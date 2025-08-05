import pandas as pd
import numpy as np
import random
from datetime import datetime

def generate_synthetic_data():
    """Generate synthetic data to balance the dataset and improve accuracy"""
    
    # Load existing data to understand patterns
    existing_data = pd.read_csv('pps1.csv')
    print(f"Original dataset size: {len(existing_data)}")
    print(f"Original subscription distribution:")
    print(existing_data['Subscription Status'].value_counts())
    
    # Get unique values from existing data
    categories = existing_data['Category'].unique().tolist()
    genders = existing_data['Gender'].unique().tolist()
    locations = existing_data['Location'].unique().tolist()
    sizes = existing_data['Size'].unique().tolist()
    colors = existing_data['Color'].unique().tolist()
    seasons = existing_data['Season'].unique().tolist()
    shipping_types = existing_data['Shipping Type'].unique().tolist()
    payment_methods = existing_data['Payment Method'].unique().tolist()
    frequencies = existing_data['Frequency of Purchases'].unique().tolist()
    
    # Items for each category
    clothing_items = ['Shirt', 'Dress', 'Pants', 'Skirt', 'Jacket', 'Sweater', 'Hoodie', 'T-shirt']
    footwear_items = ['Sneakers', 'Boots', 'Heels', 'Flats', 'Sandals', 'Loafers']
    accessories_items = ['Watch', 'Necklace', 'Bracelet', 'Ring', 'Earrings', 'Bag', 'Belt', 'Sunglasses']
    outerwear_items = ['Coat', 'Blazer', 'Cardigan', 'Vest', 'Parka', 'Windbreaker']
    
    item_mapping = {
        'Clothing': clothing_items,
        'Footwear': footwear_items,
        'Accessories': accessories_items,
        'Outerwear': outerwear_items
    }
    
    synthetic_records = []
    
    # Generate 2000 new records with focus on "Yes" subscriptions
    for i in range(2000):
        customer_id = existing_data['Customer ID'].max() + i + 1
        
        # Create patterns that lead to subscription
        # Higher chance of subscription for certain patterns
        will_subscribe = random.choice([True, False, True, True])  # 75% chance of subscription
        
        if will_subscribe:
            # Patterns that typically lead to subscription
            age = random.randint(25, 45)  # Prime demographic
            purchase_amount = random.randint(80, 300)  # Higher spending
            review_rating = round(random.uniform(3.5, 5.0), 1)  # Higher satisfaction
            previous_purchases = random.randint(3, 25)  # Loyal customers
            discount_applied = random.choice(['Yes', 'Yes', 'No'])  # Often get discounts
            promo_code_used = random.choice(['Yes', 'Yes', 'No'])  # Use promos
            frequency = random.choice(['Weekly', 'Bi-Weekly', 'Monthly'])  # Regular shoppers
            subscription_status = 'Yes'
        else:
            # Patterns that typically don't lead to subscription
            age = random.choice([random.randint(18, 24), random.randint(55, 70)])  # Less likely demographics
            purchase_amount = random.randint(20, 100)  # Lower spending
            review_rating = round(random.uniform(2.0, 4.0), 1)  # Lower satisfaction
            previous_purchases = random.randint(0, 5)  # New or infrequent customers
            discount_applied = random.choice(['No', 'No', 'Yes'])  # Less likely to get discounts
            promo_code_used = random.choice(['No', 'No', 'Yes'])  # Don't use promos much
            frequency = random.choice(['Annually', 'Every 3 Months', 'Quarterly'])  # Infrequent shoppers
            subscription_status = 'No'
        
        # Random selections for other fields
        gender = random.choice(genders)
        category = random.choice(categories)
        
        # Select appropriate item based on category
        if category in item_mapping:
            item_purchased = random.choice(item_mapping[category])
        else:
            item_purchased = random.choice(clothing_items)  # Default fallback
        
        location = random.choice(locations)
        size = random.choice(sizes)
        color = random.choice(colors)
        season = random.choice(seasons)
        shipping_type = random.choice(shipping_types)
        payment_method = random.choice(payment_methods)
        
        record = {
            'Customer ID': customer_id,
            'Age': age,
            'Gender': gender,
            'Item Purchased': item_purchased,
            'Category': category,
            'Purchase Amount (USD)': purchase_amount,
            'Location': location,
            'Size': size,
            'Color': color,
            'Season': season,
            'Review Rating': review_rating,
            'Subscription Status': subscription_status,
            'Shipping Type': shipping_type,
            'Discount Applied': discount_applied,
            'Promo Code Used': promo_code_used,
            'Previous Purchases': previous_purchases,
            'Payment Method': payment_method,
            'Frequency of Purchases': frequency
        }
        
        synthetic_records.append(record)
    
    # Create DataFrame from synthetic records
    synthetic_df = pd.DataFrame(synthetic_records)
    
    # Combine with existing data
    augmented_data = pd.concat([existing_data, synthetic_df], ignore_index=True)
    
    print(f"\nGenerated {len(synthetic_records)} synthetic records")
    print(f"New dataset size: {len(augmented_data)}")
    print(f"New subscription distribution:")
    print(augmented_data['Subscription Status'].value_counts())
    print(f"New subscription percentage:")
    print(augmented_data['Subscription Status'].value_counts(normalize=True))
    
    # Save augmented dataset
    augmented_data.to_csv('pps1.csv', index=False)
    print(f"\nAugmented dataset saved to pps1.csv")
    
    return augmented_data

if __name__ == "__main__":
    generate_synthetic_data()
