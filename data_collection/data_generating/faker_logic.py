# data_collection/data_generation/generate_feedback.py

import pandas as pd
import random
from faker import Faker

fake = Faker()

# Sample companies
event_companies = ["Live Nation", "EventBrite", "Concerts Unlimited", "Festival Mania", "Big Ticket Events"]
hotel_companies = ["Hilton Hotels", "Marriott", "Hyatt", "Sheraton", "Four Seasons"]
tour_companies = ["Globe Trekkers", "Adventure Awaits", "City Explorer", "Nature Escapes", "Wanderlust Tours"]
concert_companies = ["Ticketmaster", "StubHub", "Live Nation", "AEG Presents", "CID Entertainment"]

# Function to generate relative date format
def generate_relative_date():
    options = [
        "6 months ago", "about a year ago", "over a year ago", "almost 2 years ago", "2 years ago", "3 months ago", "almost a year ago",
        "over 2 years ago", "5 months ago", "1.5 years ago", "4 months ago", "7 months ago"
    ]
    return random.choice(options)

# Generate dynamic human-like review function
def generate_dynamic_review(sentiment, generated_reviews):
    while True:
        review_parts = []
        num_parts = random.randint(1, 3)  # Generate 1 to 3 sentence parts

        for _ in range(num_parts):
            if sentiment == "positive":
                part = random.choice([
                    "Had an absolutely amazing time! Everything was perfect.",
                    "Best experience ever! I can't wait to do this again.",
                    "Loved every moment of it! The staff was so friendly and helpful.",
                    "Such a fantastic experience. Everything was well-organized.",
                    "Great vibes, great people, and an unforgettable experience!",
                    f"The {random.choice(['amazing', 'fantastic', 'wonderful', 'great', 'excellent', 'enjoyable', 'delightful', 'wowww', 'GRAPEEE', 'Crazy good'])} {random.choice(['experience', 'event', 'time', 'performance', 'venue', 'stay', 'trip'])} was beyond my expectations."
                ])
            elif sentiment == "neutral":
                part = random.choice([
                    "It was okay, nothing too special.",
                    "Decent experience, but I expected more.",
                    "Not bad, but nothing really stood out.",
                    "An average experience, could be better.",
                    "It was fine, but I wouldn’t go out of my way to do it again.",
                    f"The {random.choice(['okay', 'decent', 'average', 'fine', 'acceptable', 'ordinary', 'adequate', 'so-so'])} {random.choice(['experience', 'event', 'time', 'performance', 'venue', 'stay', 'trip'])} didn't really impress me."
                ])
            else:  # negative
                part = random.choice([
                    "Really disappointed. It was not worth the money.",
                    "Bad experience overall. Wouldn’t recommend it.",
                    "The whole thing was a mess. Terribly organized.",
                    "Worst experience I’ve had in a long time.",
                    "Honestly, I regret going. Total letdown.",
                    f"The {random.choice(['terrible', 'awful', 'horrible', 'bad', 'disappointing', 'unpleasant', 'dreadful', 'boring'])} {random.choice(['experience', 'event', 'time', 'performance', 'venue', 'stay', 'trip'])} left me feeling frustrated."
                ])
            review_parts.append(part)

        review = " ".join(review_parts)
        if review not in generated_reviews:
            generated_reviews.add(review)
            return review

# Generate feedback data with human-like reviews
def generate_feedback_data(category, companies, organizer_type, num_samples=500):
    data = []
    generated_combinations = set()
    generated_reviews = set()

    while len(data) < num_samples:
        company = random.choice(companies)
        traveler = fake.name()
        date = generate_relative_date()
        rating = round(random.uniform(1.0, 5.0), 1)

        sentiment = "positive" if rating >= 4 else "neutral" if rating == 3 else "negative"

        review = generate_dynamic_review(sentiment, generated_reviews)
        combination = (company, traveler, date, rating, review, organizer_type)

        if combination not in generated_combinations:
            generated_combinations.add(combination)
            data.append(list(combination))

    return data

# Combine all feedback data
def generate_all_feedback():
    event_data = generate_feedback_data("Event", event_companies, "Event", 700)
    hotel_data = generate_feedback_data("Hotel", hotel_companies, "Hotel", 700)
    tour_data = generate_feedback_data("Tour", tour_companies, "Tour", 600)
    concert_data = generate_feedback_data("Concert", concert_companies, "Concert", 500)

    all_data = event_data + hotel_data + tour_data + concert_data

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=["Company Name", "User", "Date", "Rating", "Review", "Type Organizer"])

    return df

# Save to CSV
def save_to_csv():
    df = generate_all_feedback()
    df.to_csv("event_feedback.csv", index=False)
    print("Event, hotel, tour, and concert feedback data generated successfully with relative dates, organizer types, and human-like reviews.")
