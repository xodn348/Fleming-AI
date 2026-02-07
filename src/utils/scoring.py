"""Quality scoring functions for academic papers."""

from typing import Any


VENUE_TIERS = {
    "tier_1": {
        "conferences": ["NeurIPS", "ICML", "ICLR", "CVPR", "ICCV", "ACL", "EMNLP", "AAAI"],
        "journals": ["Nature", "Science", "JMLR", "TPAMI", "TACL"],
        "citation_multiplier": 0.5,
    },
    "tier_2": {
        "conferences": ["ECCV", "NAACL", "CoRL", "ICRA", "RSS", "UAI", "AISTATS"],
        "journals": ["Neural Computation", "AIJ", "MLJ"],
        "citation_multiplier": 0.75,
    },
    "tier_3": {
        "conferences": ["IJCAI", "WACV", "COLING"],
        "journals": [],
        "citation_multiplier": 1.0,
    },
}


def calculate_citation_threshold(paper_year: int, current_year: int = 2026) -> int:
    """
    Calculate the minimum citations required for 'great paper' status.

    Uses age-normalized thresholds:
    - Papers <= 1 year old: 0 citations required
    - Papers 2-3 years old: 30-90 citations (30 per year)
    - Papers 4-5 years old: 100-180 citations (40 per year after year 3)
    - Papers > 5 years old: 200+ citations (50 per year after year 5)

    Args:
        paper_year: Publication year of the paper
        current_year: Current year for age calculation (default: 2026)

    Returns:
        Minimum citation count required for 'great paper' status
    """
    age = current_year - paper_year

    if age <= 1:
        return 0
    elif age <= 3:
        return age * 30
    elif age <= 5:
        return 100 + (age - 3) * 40
    else:
        return 200 + (age - 5) * 50


def get_venue_tier(venue_name: str) -> str | None:
    """
    Determine the tier of a publication venue.

    Checks both conference and journal lists across all tiers.

    Args:
        venue_name: Name of the conference or journal

    Returns:
        Tier level as string ("tier_1", "tier_2", "tier_3") or None if not found
    """
    for tier_level, tier_data in VENUE_TIERS.items():
        conferences = tier_data.get("conferences", [])
        journals = tier_data.get("journals", [])
        if isinstance(conferences, list) and isinstance(journals, list):
            if venue_name in conferences or venue_name in journals:
                return tier_level

    return None


def calculate_quality_score(paper: dict[str, Any]) -> float:
    """
    Calculate composite quality score for a paper (0-100 scale).

    Weighted formula:
    - Velocity (40%): Citation growth rate relative to threshold
    - Venue (25%): Publication venue tier quality
    - Influence (20%): Citation count relative to threshold
    - Awards (10%): Award/recognition count
    - Concepts (5%): Number of unique concepts/keywords

    Args:
        paper: Dictionary containing paper metadata with keys:
            - year: Publication year (int)
            - citations: Citation count (int)
            - venue: Publication venue name (str)
            - awards: Number of awards (int, optional)
            - concepts: List of concepts/keywords (list, optional)

    Returns:
        Quality score between 0 and 100
    """
    current_year = 2026

    # Extract paper attributes
    paper_year = paper.get("year", current_year)
    citations = paper.get("citations", 0)
    venue = paper.get("venue", "")
    awards = paper.get("awards", 0)
    concepts = paper.get("concepts", [])

    # 1. Velocity score (40%) - citation growth rate
    age = current_year - paper_year
    if age <= 0:
        velocity_score = 0
    else:
        citations_per_year = citations / age
        # Normalize: assume 10 citations/year is excellent
        velocity_score = min(100, (citations_per_year / 10) * 100)

    # 2. Venue score (25%) - tier-based quality
    venue_tier = get_venue_tier(venue)
    if venue_tier == "tier_1":
        venue_score = 100
    elif venue_tier == "tier_2":
        venue_score = 75
    elif venue_tier == "tier_3":
        venue_score = 50
    else:
        venue_score = 25  # Unknown venue

    # 3. Influence score (20%) - citations vs threshold
    citation_threshold = calculate_citation_threshold(paper_year, current_year)
    if citation_threshold == 0:
        influence_score = min(100, citations * 10)  # New papers: 10 points per citation
    else:
        influence_score = min(100, (citations / citation_threshold) * 100)

    # 4. Awards score (10%) - recognition count
    awards_score = min(100, awards * 20)  # 5 awards = 100 points

    # 5. Concepts score (5%) - unique concepts
    concept_count = len(concepts) if isinstance(concepts, list) else 0
    concepts_score = min(100, concept_count * 10)  # 10 concepts = 100 points

    # Weighted composite score
    quality_score = (
        (velocity_score * 0.40)
        + (venue_score * 0.25)
        + (influence_score * 0.20)
        + (awards_score * 0.10)
        + (concepts_score * 0.05)
    )

    return round(quality_score, 2)
