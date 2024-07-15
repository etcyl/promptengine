def process_numbers(numbers):
    """
    This function takes a list of numbers and returns a new list that contains:
    - "even" if the number is even
    - "odd" if the number is odd
    - "zero" if the number is zero
    """
    result = []
    for number in numbers:
        if number == 0:
            result.append("zero")
        elif number % 2 == 0:
            result.append("even")
        else:
            result.append("odd")
    return result

# Example usage
if __name__ == "__main__":
    sample_numbers = [0, 1, 2, 3, 4, 5, -1, -2]
    print(process_numbers(sample_numbers))
