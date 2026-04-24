#!/usr/bin/env python3
"""Generate TPC-H parquet data in Hive-style directory layout with Velox-compatible schema.

DuckDB's dbgen produces DECIMAL(15,2) and INTEGER types, but Velox expects DOUBLE
and BIGINT. This script casts all columns to match Velox's TpchGen.cpp schema exactly.

Usage:
    python gen_tpch_parquet.py [--sf 100] [--output /path/to/output]

If --output is not specified, defaults to <repo>/test_datasets/tpch/sf<SF>
"""

import argparse
import duckdb
import os
import time

TABLES = [
    "lineitem",
    "orders",
    "customer",
    "part",
    "partsupp",
    "supplier",
    "nation",
    "region",
]

def main():
    parser = argparse.ArgumentParser(description="Generate TPC-H parquet data with Velox schema")
    parser.add_argument("--sf", type=int, default=100, help="Scale factor (default: 100)")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--memory", type=str, default="64GB", help="DuckDB memory limit")
    parser.add_argument("--threads", type=int, default=0, help="DuckDB threads (0=auto)")
    args = parser.parse_args()

    sf = args.sf
    if args.output:
        output_dir = args.output
    else:
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(repo_root, "test_datasets", "tpch", f"sf{sf}")

    print(f"=== Generating TPC-H SF{sf} Parquet Data ===")
    print(f"Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    con = duckdb.connect()
    con.execute("INSTALL tpch; LOAD tpch;")
    con.execute(f"SET memory_limit='{args.memory}';")
    if args.threads > 0:
        con.execute(f"SET threads={args.threads};")
    con.execute("SET temp_directory='{}';".format(os.path.join(output_dir, "_tmp")))
    
    print(f"Generating TPC-H SF{sf} (memory={args.memory})...")
    t0 = time.time()
    con.execute(f"CALL dbgen(sf={sf})")
    print(f"  dbgen done in {time.time() - t0:.1f}s")

    # Exact target schema from Velox's TpchGen.cpp to avoid type mismatch errors.
    # Velox doesn't do implicit casts (e.g. multiply(DOUBLE,INTEGER) fails).
    VELOX_SCHEMA = {
        "lineitem": {
            "l_orderkey": "BIGINT", "l_partkey": "BIGINT", "l_suppkey": "BIGINT",
            "l_linenumber": "INTEGER", "l_quantity": "DOUBLE",
            "l_extendedprice": "DOUBLE", "l_discount": "DOUBLE", "l_tax": "DOUBLE",
            "l_returnflag": None, "l_linestatus": None,
            "l_shipdate": None, "l_commitdate": None, "l_receiptdate": None,
            "l_shipinstruct": None, "l_shipmode": None, "l_comment": None,
        },
        "orders": {
            "o_orderkey": "BIGINT", "o_custkey": "BIGINT", "o_orderstatus": None,
            "o_totalprice": "DOUBLE", "o_orderdate": None, "o_orderpriority": None,
            "o_clerk": None, "o_shippriority": "INTEGER", "o_comment": None,
        },
        "customer": {
            "c_custkey": "BIGINT", "c_name": None, "c_address": None,
            "c_nationkey": "BIGINT", "c_phone": None, "c_acctbal": "DOUBLE",
            "c_mktsegment": None, "c_comment": None,
        },
        "part": {
            "p_partkey": "BIGINT", "p_name": None, "p_mfgr": None,
            "p_brand": None, "p_type": None, "p_size": "INTEGER",
            "p_container": None, "p_retailprice": "DOUBLE", "p_comment": None,
        },
        "partsupp": {
            "ps_partkey": "BIGINT", "ps_suppkey": "BIGINT",
            # ps_availqty is INTEGER in Velox schema but used in arithmetic with
            # DOUBLE (ps_supplycost * ps_availqty in Q11), so must be DOUBLE too.
            "ps_availqty": "DOUBLE", "ps_supplycost": "DOUBLE", "ps_comment": None,
        },
        "supplier": {
            "s_suppkey": "BIGINT", "s_name": None, "s_address": None,
            "s_nationkey": "BIGINT", "s_phone": None, "s_acctbal": "DOUBLE",
            "s_comment": None,
        },
        "nation": {
            "n_nationkey": "BIGINT", "n_name": None,
            "n_regionkey": "BIGINT", "n_comment": None,
        },
        "region": {
            "r_regionkey": "BIGINT", "r_name": None, "r_comment": None,
        },
    }

    for table in TABLES:
        table_dir = os.path.join(output_dir, table)
        os.makedirs(table_dir, exist_ok=True)
        out_path = os.path.join(table_dir, f"{table}.parquet")

        if os.path.exists(out_path):
            sz = os.path.getsize(out_path) / (1024 * 1024)
            print(f"  {table}: already exists ({sz:.1f} MB), skipping")
            continue

        print(f"  Exporting {table}...", end="", flush=True)
        t1 = time.time()
        # Build column list, casting to match Velox's expected schema
        cols_info = con.execute(f"DESCRIBE {table}").fetchall()
        target_types = VELOX_SCHEMA[table]
        col_exprs = []
        for col_name, col_type, *_ in cols_info:
            target = target_types.get(col_name)
            if target and target.upper() != col_type.upper():
                col_exprs.append(f"CAST({col_name} AS {target}) AS {col_name}")
            else:
                col_exprs.append(col_name)
        select_clause = ", ".join(col_exprs)
        con.execute(f"COPY (SELECT {select_clause} FROM {table}) TO '{out_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        elapsed = time.time() - t1
        
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f" {size_mb:.1f} MB in {elapsed:.1f}s")

    con.close()

    # Print summary
    total = 0
    print("\n=== Summary ===")
    for table in TABLES:
        p = os.path.join(output_dir, table, f"{table}.parquet")
        if os.path.exists(p):
            sz = os.path.getsize(p) / (1024 * 1024)
            total += sz
            print(f"  {table:12s} {sz:8.1f} MB")
    print(f"  {'TOTAL':12s} {total:8.1f} MB ({total/1024:.1f} GB)")
    print(f"\nData ready at: {output_dir}")

if __name__ == "__main__":
    main()
