from summarizer import summarize_text

sample_text = """
This Agreement is made on this 23rd day of June, 2026, by and between Licensor and Licensee.
Whereas, the parties desire to enter into a business relationship and cooperate on technical projects.
Now, therefore, the Licensor hereby grants to the Licensee a non-exclusive, non-transferable, revocable license to use the Software.
Licensee shall pay the Licensor a monthly royalty fee of $100 USD for the duration of this agreement.
Either party may terminate this agreement at any time by giving 30 days written notice to the other party.
This contract shall be governed by, and construed in accordance with, the laws of the State of New York.
In witness whereof, the parties hereto have executed this Agreement as of the date first written above.
"""

print("--- Source Text ---")
print(sample_text.strip())
print("\n--- Summary (3 sentences) ---")
try:
    summary = summarize_text(sample_text, target_count=3)
    for idx, sent in enumerate(summary):
        print(f"{idx+1}. {sent}")
except Exception as e:
    print(f"Error during summarization: {e}")
