"""Azure AI Search: add vectors and compare keyword, vector, hybrid, and hybrid + semantic ranker.

Setup (once):
    pip install azure-search-documents openai azure-identity

Env vars (terminal only, never in the file):
    SEARCH_ENDPOINT              https://<your-service>.search.windows.net
    SEARCH_ADMIN_KEY             admin key from the portal (Settings > Keys)
    AZURE_OPENAI_ENDPOINT        optional; defaults to CONTENTUNDERSTANDING_ENDPOINT (your cognitiveservices.azure.com endpoint)
    AZURE_EMBEDDING_DEPLOYMENT   optional; defaults to text-embedding-3-small

Run:
    python search_hybrid_demo.py
    python search_hybrid_demo.py --kb kb.json         # your own documents: JSON list of {"id","title","content","category"}
    python search_hybrid_demo.py --query "your own question"
    python search_hybrid_demo.py --cleanup            # delete this index (free tier allows only 3)
"""
import argparse
import json
import os
import sys
import time

import openai
from azure.core.credentials import AzureKeyCredential
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)
from azure.search.documents.models import VectorizedQuery

SEARCH_ENDPOINT = os.environ["SEARCH_ENDPOINT"]
SEARCH_KEY = os.environ["SEARCH_ADMIN_KEY"]
AOAI_ENDPOINT = os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ["CONTENTUNDERSTANDING_ENDPOINT"]
EMB_DEPLOYMENT = os.environ.get("AZURE_EMBEDDING_DEPLOYMENT", "text-embedding-3-small")
DIMENSIONS = 1536  # text-embedding-3-small returns 1536 numbers per text
INDEX = "faq-index-vec"
SEMANTIC_CONFIG = "faq-semantic"

SAMPLE_DOCS = [
    {"id": "1", "category": "account", "title": "Resetting your password",
     "content": "If you can no longer sign in, choose Forgot password on the login page. We email a reset link that expires after 30 minutes."},
    {"id": "2", "category": "billing", "title": "Refund policy",
     "content": "You can request your money back within 14 days of purchase. Refunds are returned to the original payment method in 5 to 10 business days."},
    {"id": "3", "category": "billing", "title": "Cancelling your subscription",
     "content": "Open Billing, then Plan, and select End subscription. You keep access until the end of the current billing period."},
    {"id": "4", "category": "performance", "title": "Dashboard loads slowly",
     "content": "Large date ranges and many widgets slow the dashboard. Narrow the date range, remove unused widgets, or enable cached reports."},
    {"id": "5", "category": "security", "title": "Two-factor authentication",
     "content": "Turn on two-factor authentication under Security settings. You can use an authenticator app or text message codes."},
    {"id": "6", "category": "account", "title": "Changing your email address",
     "content": "Go to Profile and edit the email field. We send a confirmation message to the new address before the change takes effect."},
    {"id": "7", "category": "data", "title": "Exporting your data",
     "content": "Use Settings, then Data, then Export. Reports are available as CSV or PDF and are emailed when ready."},
    {"id": "8", "category": "billing", "title": "Updating a credit card",
     "content": "Update the card under Billing, then Payment methods. Failed payments are retried automatically for three days."},
    {"id": "9", "category": "team", "title": "Adding teammates",
     "content": "Invite colleagues from Team settings. Each invited person gets a role: viewer, editor or admin."},
    {"id": "10", "category": "security", "title": "Locked account",
     "content": "After five wrong attempts your account is locked for 15 minutes. Wait, or contact support to unlock it sooner."},
    {"id": "11", "category": "integrations", "title": "Connecting Slack",
     "content": "Install the Slack app from Integrations and pick a channel. Alerts are posted there when a report finishes."},
    {"id": "12", "category": "data", "title": "Deleting your account",
     "content": "Account deletion removes all stored reports after 30 days. This cannot be undone once the period has passed."},
]

DEFAULT_QUERIES = [
    "I forgot how to get into my account",
    "get my money back",
    "stop paying every month",              
    "why is the screen taking forever",
    "how do I stop being charged",
]


def embed(texts):
    """Turn a list of texts into a list of vectors using your Azure embedding deployment (keyless sign-in)."""
    token = get_bearer_token_provider(DefaultAzureCredential(), "https://cognitiveservices.azure.com/.default")
    client = openai.AzureOpenAI(
        azure_endpoint=AOAI_ENDPOINT,
        azure_ad_token_provider=token,
        api_version="2025-04-01-preview",
        max_retries=3,
    )
    try:
        resp = client.embeddings.create(model=EMB_DEPLOYMENT, input=texts)
    except openai.RateLimitError as e:
        sys.exit(f"Rate limited on the embedding deployment. Wait a minute and retry.\n{e}")
    return [d.embedding for d in resp.data]


def build_index():
    client = SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY))
    index = SearchIndex(
        name=INDEX,
        fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SearchField(
                name="content_vector",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=DIMENSIONS,
                vector_search_profile_name="vec-profile",
            ),
        ],
        vector_search=VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="vec-profile", algorithm_configuration_name="hnsw")],
        ),
        semantic_search=SemanticSearch(
            configurations=[
                SemanticConfiguration(
                    name=SEMANTIC_CONFIG,
                    prioritized_fields=SemanticPrioritizedFields(
                        title_field=SemanticField(field_name="title"),
                        content_fields=[SemanticField(field_name="content")],
                    ),
                )
            ]
        ),
    )
    client.create_or_update_index(index)
    print(f"Index '{INDEX}' ready (text fields + a {DIMENSIONS}-number vector field).")


def upload(docs):
    texts = [f"{d['title']}. {d['content']}" for d in docs]
    vectors = embed(texts)
    prepared = [{**d, "content_vector": v} for d, v in zip(docs, vectors)]
    client = SearchClient(SEARCH_ENDPOINT, INDEX, AzureKeyCredential(SEARCH_KEY))
    result = client.upload_documents(documents=prepared)
    print(f"Uploaded {sum(1 for r in result if r.succeeded)} of {len(docs)} documents, each with its vector.")
    time.sleep(3)  # indexing takes a moment
    return client


def show(label, results):
    if not results:
        print(f"  {label}: no results")
        return
    print(f"  {label}")
    for r in results:
        rerank = r.get("@search.reranker_score")
        extra = f"  reranker={rerank:.2f}" if rerank is not None else ""
        print(f"    - {r['title']}  (score={r['@search.score']:.3f}{extra})")


def compare(client, query):
    print(f"\nQUERY: {query}")
    qvec = embed([query])[0]
    vq = VectorizedQuery(vector=qvec, k_nearest_neighbors=5, fields="content_vector")
    modes = [
        ("1. keyword only (BM25)", dict(search_text=query)),
        ("2. vector only (meaning)", dict(search_text=None, vector_queries=[vq])),
        ("3. hybrid: keyword + vector, merged by RRF", dict(search_text=query, vector_queries=[vq])),
        ("4. hybrid + semantic ranker",
         dict(search_text=query, vector_queries=[vq], query_type="semantic",
              semantic_configuration_name=SEMANTIC_CONFIG, query_caption="extractive")),
    ]
    for label, kw in modes:
        show(label, list(client.search(top=3, select=["title"], **kw)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", help="path to your own documents (JSON list)")
    ap.add_argument("--query", help="run one query of your own")
    ap.add_argument("--cleanup", action="store_true", help="delete the index and exit")
    args = ap.parse_args()

    if args.cleanup:
        SearchIndexClient(SEARCH_ENDPOINT, AzureKeyCredential(SEARCH_KEY)).delete_index(INDEX)
        print(f"Deleted index '{INDEX}'.")
        return

    docs = SAMPLE_DOCS
    if args.kb:
        with open(args.kb, encoding="utf-8") as f:
            docs = json.load(f)

    build_index()
    client = upload(docs)
    print("Scores are not comparable across modes: BM25 scores, vector similarity and RRF scores use different scales. Compare the ORDER.")
    for q in ([args.query] if args.query else DEFAULT_QUERIES):
        compare(client, q)


if __name__ == "__main__":
    main()