import argparse
from collections import defaultdict
from itertools import combinations, pairwise, permutations
from multiprocessing import Pool
import time

import numpy as np
from sklearn.datasets import fetch_20newsgroups
import tqdm

SHINGLE_SIZE = 4
NUM_PERMUTATIONS = 128
BAND_SIZE = 4
PROCESSES = 1
BATCH_SIZE = 2500


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Locality-Sensitive Hashing for document similarity detection"
    )

    # Optional arguments with defaults matching module constants
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Use the full dataset instead of a subset",
    )
    parser.add_argument(
        "--pairwise-demo",
        action="store_true",
        help="Run a pairwise similarity demo on a subset of the dataset",
    )
    parser.add_argument(
        "--shingle-size",
        type=int,
        default=SHINGLE_SIZE,
        help=f"Size of shingles (default: {SHINGLE_SIZE})",
    )
    parser.add_argument(
        "--num-permutations",
        type=int,
        default=NUM_PERMUTATIONS,
        help=f"Number of permutations for minhash (default: {NUM_PERMUTATIONS})",
    )
    parser.add_argument(
        "--band-size",
        type=int,
        default=BAND_SIZE,
        help=f"Size of bands for LSH (default: {BAND_SIZE})",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=PROCESSES,
        help=f"Number of parallel processes (default: {PROCESSES})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Number of documents per batch (default: {BATCH_SIZE})",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for pairs (default: 0.8)",
    )

    args = parser.parse_args()

    if args.shingle_size <= 0:
        parser.error("shingle_size must be positive")
    if args.shingle_size > 6:
        input(
            "WARNING: Setting a shingle size greater than 6 may lead to high memory usage. Press Enter to continue..."
        )
    if args.num_permutations <= 0:
        parser.error("num_permutations must be positive")
    if args.band_size <= 0:
        parser.error("band_size must be positive")
    if args.num_permutations % args.band_size != 0:
        parser.error("num_permutations must be divisible by band_size")
    if args.processes <= 0:
        parser.error("processes must be positive")
    if args.batch_size <= 0:
        parser.error("batch_size must be positive")
    if not 0 <= args.similarity_threshold <= 1:
        parser.error("similarity_threshold must be between 0 and 1")

    return args


def get_shingles(text: str, k: int = 3) -> list[str]:
    shingles = set()
    for i in range(len(text) - (k + 1)):
        shingles.add(text[i : i + k])
    return list(shingles)


def get_vocabulary(corpus: list[str], k: int = 3) -> list[str]:
    vocabulary = set()
    for row in corpus:
        vocabulary.update(get_shingles(row, k))

    return list(vocabulary)


def get_permutations(vocab_len, num_permutations):
    m = []
    for seed in range(num_permutations):
        rng = np.random.default_rng(seed)
        m.append(rng.integers(low=0, high=vocab_len, size=vocab_len))
    return np.array(m)


def get_minhash(text, shingle_to_idx, permutations, shingle_size):
    idx = np.array([shingle_to_idx[s] for s in get_shingles(text, shingle_size)])
    return permutations[:, idx].min(axis=1)


def get_potential_pairs(document_minhashes, band_size=4):
    bands = defaultdict(list)
    for document_idx, document_minhash in enumerate(document_minhashes):
        for i in range(0, len(document_minhash), band_size):
            band_signature = tuple(document_minhash[i : i + band_size])
            bands[band_signature].append(document_idx)

    return [v for _, v in bands.items() if len(v) > 1]


_shingle_to_idx = None
_permutations = None
_shingle_size = None


def _init_worker(shingle_to_idx, permutations, shingle_size):
    global _shingle_to_idx, _permutations, _shingle_size
    _shingle_to_idx = shingle_to_idx
    _permutations = permutations
    _shingle_size = shingle_size


def process_batch(docs):
    results = []
    for text in tqdm.tqdm(docs):
        results.append(get_minhash(text, _shingle_to_idx, _permutations, _shingle_size))
    return results


def lsh_detect_pairs(
    corpus, shingle_size=3, num_permutations=128, band_size=4, processes=1, batch_size=100
):
    assert num_permutations % band_size == 0, "num_permutations must be divisible by band_size"

    vocabulary = get_vocabulary(corpus, shingle_size)
    shingle_to_idx = {shingle: idx for idx, shingle in enumerate(vocabulary)}

    permutations = get_permutations(len(vocabulary), num_permutations)
    if processes == 1:
        _init_worker(shingle_to_idx, permutations, shingle_size)
        signatures = process_batch(corpus)
    else:
        batches = [corpus[i : i + batch_size] for i in range(0, len(corpus), batch_size)]
        with Pool(
            processes,
            initializer=_init_worker,
            initargs=(shingle_to_idx, permutations, shingle_size),
        ) as pool:
            results = pool.map(process_batch, batches)
        signatures = [sig for batch in results for sig in batch]

    potential_pairs = get_potential_pairs(np.array(signatures), band_size)

    print(f"Found {len(potential_pairs)} potential sets of similar documents")

    return potential_pairs


def get_document_pair_jaccard_similarity(document_a, document_b, shingle_size=3):
    doc_a_shingles = set(get_shingles(document_a, shingle_size))
    doc_b_shingles = set(get_shingles(document_b, shingle_size))

    return len(doc_a_shingles & doc_b_shingles) / len(doc_a_shingles | doc_b_shingles)


def get_most_similar_documents(corpus, potential_pairs, shingle_size=4, minimum_similarity=0.5):
    jaccard_similarities = {}
    n_comparisons = 0
    for groups in tqdm.tqdm(potential_pairs):
        for pair in combinations(groups, 2):
            n_comparisons += 1
            document_a = corpus[pair[0]]
            document_b = corpus[pair[1]]
            similarity = get_document_pair_jaccard_similarity(
                document_a, document_b, shingle_size=shingle_size
            )
            if similarity > minimum_similarity:
                jaccard_similarities[pair] = similarity

    print(
        f"Performed {n_comparisons} comparisons and found {len(jaccard_similarities)} similar documents"
    )

    return sorted(jaccard_similarities.items(), key=lambda item: item[1], reverse=True)


def similarity_stats(similar_documents):
    scores = np.array([similarity for _, similarity in similar_documents])
    bins = [0.0] + [n / 100 for n in range(50, 105, 5)]
    counts = np.histogram(scores, bins=bins)[0]
    print("Similarity Scores:")
    for bin_start, count in zip(bins[:0:-1], counts[:0:-1]):
        print(f" Between {bin_start:0.2f} and {bin_start:0.2f}:", count)


def preview_similar_documents(corpus, similar_documents, n=10):
    for idx, (pair, similarity) in enumerate(similar_documents):
        if idx > n:
            break
        document_a = corpus[pair[0]]
        document_b = corpus[pair[1]]

        print("=" * 95)
        print("=" * 30, f"Document Pair (similarity: {similarity:0.3f})", "=" * 30)
        print("=" * 95)
        print(document_a[:1000])
        print("-" * 60)
        print(document_b[:1000])

        if input("Press any key to see the next pair, or Q to quit: ").lower() == "q":
            break


def main(args):
    if args.full_dataset:
        categories = None
    else:
        categories = ["sci.space", "sci.crypt", "sci.electronics", "sci.med"]

    corpus = fetch_20newsgroups(categories=categories).data
    print(f"Total Corpus Size: {len(corpus):,} - {len(corpus)**2:,} pairwise compairisons.")

    if args.pairwise_demo:
        print("Demoing how long a full pairwise comparison would take...")
        for i, (doc_a, doc_b) in tqdm.tqdm(
            enumerate(permutations(corpus, 2)), total=len(corpus) ** 2
        ):
            if i > 30000:
                break
            get_document_pair_jaccard_similarity(doc_a, doc_b, args.shingle_size)

    print("Starting LSH...")
    start = time.time()
    potential_pairs = lsh_detect_pairs(
        corpus,
        shingle_size=args.shingle_size,
        num_permutations=args.num_permutations,
        band_size=args.band_size,
        processes=args.processes,
        batch_size=args.batch_size,
    )
    end = time.time()
    print(f"LSH took {end-start:0.2f}s to find potential duplicate documents")

    similar_documents = get_most_similar_documents(
        corpus, potential_pairs, shingle_size=args.shingle_size, minimum_similarity=0.5
    )
    similarity_stats(similar_documents)

    if input("Preview Similar Documents? (Y/n) ").lower() == "y":
        preview_similar_documents(corpus, similar_documents)


if __name__ == "__main__":
    main(_parse_args())
