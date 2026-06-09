"""
Tools for the Detector Agent to use.
In a full implementation using Google GenAI / Agent Builder, these would be formatted as function declarations.
"""

def fetch_reviews(product_id: str):
    """Fetches reviews for a given product."""
    import sqlite3
    import os
    db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'crucible.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT id, text, rating, date, reviewer, verified_purchase FROM reviews WHERE product_id=?", (product_id,))
    reviews = cursor.fetchall()
    conn.close()
    return reviews

def linguistic_check(text: str):
    """Checks for linguistic deception cues like superlative density or mismatch."""
    text_lower = text.lower()
    superlatives = ['amazing', 'best', 'incredible', 'fantastic', 'luxurious', 'perfectly', 'life']
    count = sum(1 for word in superlatives if word in text_lower)
    if count >= 2:
        return "High superlative density detected (common in fake reviews)."
    return "Normal linguistic patterns."

def burst_check(date: str, product_id: str):
    """Checks if there was a flood of reviews appearing at once."""
    return "No significant bursts detected."

def reviewer_profile(reviewer_id: str):
    """Flags throwaway accounts."""
    if reviewer_id.startswith("bot") or "spam" in reviewer_id or "fake" in reviewer_id:
        return "Suspicious throwaway account pattern."
    return "Account appears established."

def issue_verdict(review_text: str, rating: int, reviewer: str):
    """
    Synthesizes the signals into a final verdict.
    This is mocked for the hackathon baseline. The real agent would use Gemini to weigh the tools.
    """
    ling_res = linguistic_check(review_text)
    prof_res = reviewer_profile(reviewer)
    
    if "Suspicious" in prof_res or "High superlative" in ling_res:
        return "fake"
    return "real"
