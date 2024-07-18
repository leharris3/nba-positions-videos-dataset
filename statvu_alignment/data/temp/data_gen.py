import itertools

def format_time(minutes, seconds, deciseconds):
    if minutes > 0:
        return f"{minutes:02d}:{seconds:02d}"
    elif seconds >= 10:
        return f"{seconds:02d}.{deciseconds}"
    else:
        return f"{seconds}.{deciseconds}"

def generate_time_values():
    times = []
    for minutes in range(12, -1, -1):
        for seconds in range(59, -1, -1):
            if minutes == 12 and seconds > 0:
                continue
            for deciseconds in range(10):
                times.append(format_time(minutes, seconds, deciseconds))
    return times

def save_to_file(times, filename):
    with open(filename, 'w') as f:
        for time in times:
            f.write(f"{time}\n")

# Generate all possible time values
all_times = generate_time_values()

# Save to file
save_to_file(all_times, 'nba_game_times.txt')

print(f"Generated {len(all_times)} time values and saved to nba_game_times.txt")