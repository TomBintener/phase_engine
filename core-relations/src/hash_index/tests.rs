use crate::numeric_id::NumericId;

use crate::{
    TupleIndex,
    table_shortcuts::{fill_table, v},
    table_spec::{ColumnId, WrappedTable},
};

use super::{ColumnIndex, Index};

/// Regression test for the multi-column `rebuild_full` path (used by the table
/// rebuild index): the bulk sort is stable and keyed on the value only, so rows
/// for a value that appears in *several* indexed columns used to be emitted in
/// column-concatenation order (unsorted), and the same (value, row) pair coming
/// from two columns of one row survived the adjacent-only dedup. The resulting
/// subsets violated the sorted/duplicate-free invariant, silently corrupting
/// binary searches (and congruence closure) in release builds.
#[test]
fn multi_column_rebuild_full_sorted_subsets() {
    // Values X_k appear in column 0 of rows 4k and 4k+3 and in column 1 of row
    // 4k+1 (non-contiguous, so the subset takes the sparse path). Values D_k
    // appear in both columns of row 4k+2 (the duplicate-pair case). 16 groups of
    // 4 rows produce 128 (value, row) pairs, enough to trigger the radix-sort
    // bulk path (>= 64 pairs).
    const GROUPS: u32 = 16;
    let x = |k: u32| v((1000 + k) as usize);
    let d = |k: u32| v((2000 + k) as usize);

    let mut rows = Vec::new();
    let mut filler = 3000u32;
    for k in 0..GROUPS {
        let mut f = || {
            filler += 1;
            v(filler as usize)
        };
        rows.push(vec![x(k), f(), f(), v(0)]);
        rows.push(vec![f(), x(k), f(), v(0)]);
        rows.push(vec![d(k), d(k), f(), v(0)]);
        rows.push(vec![x(k), f(), f(), v(0)]);
    }

    let table = WrappedTable::new(fill_table(
        rows,
        2,
        Some(ColumnId::new(3)),
        |old, new| {
            assert_eq!(old, new, "no conflicts in this test");
            None
        },
    ));

    use super::IndexBase;

    let mut index = ColumnIndex::new();
    let all = table.all();
    index.rebuild_full(
        &[ColumnId::new(0), ColumnId::new(1)],
        table.as_ref(),
        all.as_ref(),
    );

    for k in 0..GROUPS {
        let base = (4 * k) as usize;

        let x_key = x(k);
        let subset = index.get_subset(&x_key).expect("X_k must be indexed");
        let mut got = Vec::new();
        crate::offsets::Offsets::offsets(&subset, |row| got.push(row.index()));
        assert_eq!(
            got,
            vec![base, base + 1, base + 3],
            "rows for X_{k} must be sorted and duplicate-free"
        );

        let d_key = d(k);
        let subset = index.get_subset(&d_key).expect("D_k must be indexed");
        let mut got = Vec::new();
        crate::offsets::Offsets::offsets(&subset, |row| got.push(row.index()));
        assert_eq!(
            got,
            vec![base + 2],
            "same (value, row) pair from two columns must be deduplicated for D_{k}"
        );
    }
}

#[test]
fn basic_updates() {
    // Get slightly higher coverage with nondeterministic parallelism.
    for _ in 0..10 {
        // fill a SortedWritesTable with some data. confirm that an index built on
        // some subset of columns works as expected. Then add more data, and confirm
        // that updates still work.
        let mut table = WrappedTable::new(fill_table(
            vec![
                vec![v(0), v(1), v(2), v(0)],
                vec![v(1), v(2), v(3), v(0)],
                vec![v(2), v(3), v(4), v(0)],
                vec![v(3), v(4), v(5), v(1)],
                vec![v(4), v(5), v(6), v(1)],
            ],
            2,
            Some(ColumnId::new(3)),
            |old, new| {
                assert_eq!(old, new, "no conflicts in this test");
                None
            },
        ));

        let mut index = Index::new(vec![ColumnId::new(0), ColumnId::new(2)], TupleIndex::new(2));
        assert!(index.get_subset(&[v(0), v(2)]).is_none());
        index.refresh(table.as_ref());
        for i in 0..=4 {
            let key = [v(i), v(i + 2)];
            let subset = index.get_subset(&key).unwrap();
            table.scan(subset).iter().for_each(|(id, row)| {
                assert_eq!(&row[0..3], &[v(i), v(i + 1), v(i + 2)]);
                let readback = table.get_row(&row[0..2]).expect("row should exist");
                assert_eq!(readback.id, id);
                assert_eq!(readback.vals.as_slice(), row);
            });
        }

        {
            let mut buf = table.new_buffer();
            for i in 5..10 {
                buf.stage_insert(&[v(i), v(i + 1), v(i + 2), v(2)]);
            }
        }

        empty_execution_state!(es);
        table.merge(&mut es);
        index.refresh(table.as_ref());
        for i in 0..10 {
            let key = [v(i), v(i + 2)];
            let subset = index.get_subset(&key).unwrap();
            table.scan(subset).iter().for_each(|(id, row)| {
                assert_eq!(&row[0..3], &[v(i), v(i + 1), v(i + 2)]);
                let readback = table.get_row(&row[0..2]).expect("row should exist");
                assert_eq!(readback.id, id);
                assert_eq!(readback.vals.as_slice(), row);
            });
        }

        // Now get an update to the major version.
        let start_version = table.version().major;
        while table.version().major == start_version {
            table.new_buffer().stage_remove(&[v(0), v(1)]);
            table.merge(&mut es);
            table.new_buffer().stage_insert(&[v(0), v(1), v(2), v(3)]);
            table.merge(&mut es);
        }

        // Refresh should do the right thing.
        index.refresh(table.as_ref());
        for i in 0..10 {
            let key = [v(i), v(i + 2)];
            let subset = index.get_subset(&key).unwrap();
            table.scan(subset).iter().for_each(|(id, row)| {
                assert_eq!(&row[0..3], &[v(i), v(i + 1), v(i + 2)]);
                let readback = table.get_row(&row[0..2]).expect("row should exist");
                assert_eq!(readback.id, id);
                assert_eq!(readback.vals.as_slice(), row);
            });
        }
    }
}
