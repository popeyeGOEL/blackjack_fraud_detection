import random
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Blackjack game simulation
def simulate_blackjack_game(card_counting=False):
    deck = [2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10, 11] * 4
    random.shuffle(deck)

    player_hand = []
    dealer_hand = []
    running_count = 0
    true_count = 0

    # Deal initial hands
    player_hand.append(deck.pop())
    dealer_hand.append(deck.pop())
    player_hand.append(deck.pop())
    dealer_hand.append(deck.pop())

    # Player's turn
    while sum(player_hand) < 17:
        if card_counting:
            # Implement card counting strategy (e.g., Hi-Lo)
            if 2 <= player_hand[-1] <= 6:
                running_count += 1
            elif player_hand[-1] >= 10:
                running_count -= 1

            if len(deck) >= 52:
                true_count = running_count // (len(deck) // 52)
            else:
                true_count = running_count  # Set true_count to running_count when deck is less than 52 cards

            if true_count > 0 and sum(player_hand) <= 11:
                if len(deck) > 0:  # Check if the deck is not empty before drawing a card
                    player_hand.append(deck.pop())
                else:
                    break  # Exit the loop if the deck is empty
            else:
                break
        else:
            # Basic strategy
            if sum(player_hand) <= 11:
                if len(deck) > 0:  # Check if the deck is not empty before drawing a card
                    player_hand.append(deck.pop())
                else:
                    break  # Exit the loop if the deck is empty
            else:
                break

    # Dealer's turn
    while sum(dealer_hand) < 17:
        if len(deck) > 0:  # Check if the deck is not empty before drawing a card
            dealer_hand.append(deck.pop())
        else:
            break  # Exit the loop if the deck is empty

    # Determine outcome
    player_sum = sum(player_hand)
    dealer_sum = sum(dealer_hand)

    if player_sum > 21:
        outcome = -1  # Player busts, lose
    elif dealer_sum > 21:
        outcome = 1  # Dealer busts, win
    elif player_sum > dealer_sum:
        outcome = 1  # Player wins
    elif player_sum < dealer_sum:
        outcome = -1  # Player loses
    else:
        outcome = 0  # Push

    # Record features
    features = {
        'player_initial_hand': player_hand[0] + player_hand[1],
        'dealer_up_card': dealer_hand[0],
        'player_actions': len(player_hand) - 2,
        'running_count': running_count,
        'true_count': true_count if len(deck) > 0 else running_count,  # Use running_count if deck is empty
        'outcome': outcome,
        'label': 1 if card_counting else 0
    }

    return features

# Simulate games
num_normal_games = 100000
num_card_counting_games = 10000

normal_games = [simulate_blackjack_game() for _ in range(num_normal_games)]
card_counting_games = [simulate_blackjack_game(card_counting=True) for _ in range(num_card_counting_games)]

# Combine data
dataset = normal_games + card_counting_games
df = pd.DataFrame(dataset)

# Preprocess data
scaler = StandardScaler()
features = ['player_initial_hand', 'dealer_up_card', 'player_actions', 'running_count', 'true_count']
df[features] = scaler.fit_transform(df[features])

# Split dataset
X = df.drop(['outcome', 'label'], axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save dataset
df.to_csv('blackjack_dataset.csv', index=False)