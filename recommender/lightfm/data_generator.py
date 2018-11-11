import random

if __name__ == "__main__":
    interaction_file = open('interaction.csv', 'wb')

    n_users = 10000
    n_items = 10000

    user_item_min = 20
    user_item_max = 50
    
    items = range(n_items)

    for user_id in xrange(n_users):
        random.shuffle(items)

        n_liked_items = random.randint(user_item_min, user_item_max)

        liked_items = items[:n_liked_items]

        for item_id in liked_items:
            is_liked = random.randint(0, 1)
            row = '{0},{1},{2}\n'.format(user_id, item_id, is_liked)
            interaction_file.write(row)

    interaction_file.close()