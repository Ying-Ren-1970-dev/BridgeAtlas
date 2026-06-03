"""Calculate API cost for enriching all projects."""
from pathlib import Path
import fitz

# GPT-4o Vision pricing (as of 2024)
# Input: $2.50 per 1M tokens
# High detail images: ~765 tokens per image + text prompt (~500 tokens)
TOKEN_COST_PER_MILLION = 2.50
TOKENS_PER_IMAGE = 765
PROMPT_TOKENS = 500

projects_folder = Path("Projects")
pdf_files = list(projects_folder.glob("*.pdf"))

print(f"Found {len(pdf_files)} PDF files in Projects folder\n")
print("=" * 80)
print(f"{'File Name':<50} {'Pages':<8} {'Sampled':<8}")
print("=" * 80)

total_pages = 0
total_sampled = 0

for pdf_file in sorted(pdf_files):
    try:
        doc = fitz.open(pdf_file)
        page_count = len(doc)
        doc.close()
        
        # Calculate sampled pages based on enrichment logic
        if page_count <= 15:
            sampled = page_count
        else:
            # First 5 + Middle 5 + Last 5
            sampled = 15
        
        total_pages += page_count
        total_sampled += sampled
        
        print(f"{pdf_file.name:<50} {page_count:<8} {sampled:<8}")
    except Exception as e:
        print(f"{pdf_file.name:<50} {'ERROR':<8} {str(e)[:30]}")

print("=" * 80)
print(f"{'TOTAL':<50} {total_pages:<8} {total_sampled:<8}")
print("=" * 80)

# Calculate API cost
total_tokens = total_sampled * (TOKENS_PER_IMAGE + PROMPT_TOKENS)
cost = (total_tokens / 1_000_000) * TOKEN_COST_PER_MILLION

print(f"\nAPI COST ESTIMATE:")
print(f"  Total pages in all PDFs:        {total_pages:,}")
print(f"  Pages to be analyzed (sampled): {total_sampled:,}")
print(f"  Tokens per image:               {TOKENS_PER_IMAGE:,}")
print(f"  Prompt tokens per request:      {PROMPT_TOKENS:,}")
print(f"  Total tokens estimated:         {total_tokens:,}")
print(f"  Cost per 1M tokens:             ${TOKEN_COST_PER_MILLION}")
print(f"\n  *** ESTIMATED TOTAL COST: ${cost:.2f} ***")

print(f"\nNOTES:")
print(f"  - Small PDFs (≤15 pages): All pages analyzed")
print(f"  - Large PDFs (>15 pages): 15 pages sampled (first 5, middle 5, last 5)")
print(f"  - This is an estimate; actual cost may vary slightly")
print(f"  - Uses GPT-4o Vision with high detail mode")
