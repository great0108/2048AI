def save_history(history, score):
    history_filename = "game_history.txt"
    with open(history_filename, "w") as f:
        f.write(f"Final Score: {score}\n")
        f.write("="*20 + "\n\n")
        f.write("--- Game History ---\n")
        for i, record in enumerate(history):
            f.write(f"Turn {i+1}: AI chose {record['move']}\n")
            for row in record['board']:
                f.write(str(row) + "\n")
            f.write("-" * 10 + "\n")
        f.write("--- End of History ---\n")
    
    print(f"Game history saved to {history_filename}")