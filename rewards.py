def scale_number(num, upper_bound, lower_bound):
    if num > upper_bound:
        return 1
    elif num < lower_bound:
        return -1
    else:
        gradient = (num - lower_bound) / (upper_bound - lower_bound)
        scaled_num = 2 * gradient - 1
        return scaled_num