"""
twitter_client.py — Conexión a Twitter/X con fallback a tweets demo
"""
import os

TWEETS_DEMO = [
    "I love how diverse our community is becoming! Everyone deserves respect.",
    "These immigrants are ruining everything, go back to your country.",
    "Happy to see more women in leadership positions across the industry!",
    "Women shouldn't be in politics, they're too emotional to lead.",
    "Celebrating our differences makes us stronger as a society.",
    "All these foreigners take our jobs and don't belong here.",
    "Had an amazing multicultural festival today, the food was incredible!",
    "Muslim terrorists are everywhere, we need to ban them all.",
    "Just saw a beautiful sunset, feeling grateful and happy today.",
    "Black people are always causing trouble in this neighborhood.",
    "Unity and respect for all cultures is the foundation of progress.",
    "Why do we keep letting these people into our country? Disgusting.",
    "Proud to work in such an inclusive environment with amazing colleagues.",
    "Go back to where you came from, nobody wants you here.",
    "Science and education are the tools to build a better tomorrow.",
]


def buscar_tweets(termino: str, max_tweets: int = 10) -> tuple[list[str], str]:
    """
    Devuelve (lista_de_tweets, fuente).
    fuente = 'twitter' | 'demo'
    """
    bearer = os.getenv("TWITTER_BEARER_TOKEN", "")

    if bearer:
        try:
            import tweepy
            client = tweepy.Client(bearer_token=bearer)
            query    = f"{termino} lang:en -is:retweet"
            response = client.search_recent_tweets(
                query=query,
                max_results=min(max_tweets, 100),
                tweet_fields=["text"]
            )
            if response.data:
                return [t.text for t in response.data], "twitter"
        except Exception as e:
            print(f"Twitter falló ({e}), usando tweets demo...")

    # Fallback demo
    return TWEETS_DEMO[:max_tweets], "demo"
