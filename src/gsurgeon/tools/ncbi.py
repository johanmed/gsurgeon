"""Modules with NCBI tools for GSurgeon"""

import json

import dspy
from Bio.Entrez import efetch, esearch, esummary, read


def search_ncbi(database: str, term: str, max_results: int = 10) -> str:
    handle = esearch(db=database, term=term, retmax=max_results)
    records = read(handle)
    handle.close()
    # Order records for determinism
    if isinstance(records, dict) and "IdList" in records:
        records["IdList"] = sorted(records["IdList"])
    return json.dumps(records, sort_keys=True)


ncbi_searcher = dspy.Tool(
    name="ncbi_searcher",
    desc="Search an NCBI database (e.g., nucleotide, protein, pubmed) for a term",
    args={
        "database": {
            "type": "string",
            "desc": "Database name like 'nucleotide' or 'pubmed'",
        },
        "term": {"type": "string", "desc": "Search term or query"},
        "max_results": {
            "type": "integer",
            "desc": "Max results (default 2000)",
            "default": 2000,
        },
    },
    func=search_ncbi,
)


def fetch_record(database: str, record_id: str, rettype: str) -> str:
    handle = efetch(db=database, id=record_id, rettype=rettype, retmode="text")
    result = handle.readline().strip()
    handle.close()
    return result


record_fetcher = dspy.Tool(
    name="record_fetcher",
    desc="Fetch a record from an NCBI database (e.g., nucleotide, protein, pubmed)",
    args={
        "database": {
            "type": "string",
            "desc": "Database name like 'nucleotide' or 'pubmed'",
        },
        "record_id": {"type": "string", "desc": "Identifier of record"},
        "rettype": {"type": "string", "desc": "Return type compatible with database"},
    },
    func=fetch_record,
)


def summarize_record(database: str, record_id: str) -> str:
    handle = esummary(db=database, id=record_id)
    result = read(handle)
    handle.close()
    # If a list of summaries, sort by Id for determinism
    if isinstance(result, list):
        result = sorted(result, key=lambda x: x.get("Id", ""))
    return json.dumps(result, sort_keys=True)


record_synthesizer = dspy.Tool(
    name="record_synthesiser",
    desc="Get summary on a record from an NCBI database (e.g., nucleotide, protein, pubmed)",
    args={
        "database": {
            "type": "string",
            "desc": "Database name like 'nucleotide' or 'pubmed'",
        },
        "record_id": {"type": "string", "desc": "Identifier of record"},
    },
    func=summarize_record,
)
