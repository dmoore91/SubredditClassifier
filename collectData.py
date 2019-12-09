import praw
import json
from random import shuffle

def store_comments_for_dankchristianmemes(reddit):
    list_of_comments = []
    fields = ('body','created_utc', 'distinguished','edited', 'id', 'is_submitter',
              'link_id', 'parent_id', 'permalink', 'replies', 'score', 'stickied', 'submission', 'subreddit_id')

    for comment in reddit.subreddit('dankchristianmemes').comments(limit=1000):
        to_dict = vars(comment)
        sub_dict = {field: to_dict.get(field) for field in fields}
        list_of_comments.append(sub_dict)

    shuffle(list_of_comments)

    with open("dcm_comments_training_data.txt", "w") as f:
        json.dump(list_of_comments[:500], f)

    with open("dcm_comments_development_data.txt", "w") as f:
        json.dump(list_of_comments[500:750], f)

    with open("dcm_comments_test_data.txt", "w") as f:
        json.dump(list_of_comments[750:1000], f)


def store_comments_for_izlam(reddit):
    list_of_comments = []
    fields = ('body','created_utc', 'distinguished','edited', 'id', 'is_submitter',
              'link_id', 'parent_id', 'permalink', 'replies', 'score', 'stickied', 'submission', 'subreddit_id')

    for comment in reddit.subreddit('Izlam').comments(limit=1000):
        to_dict = vars(comment)
        sub_dict = {field: to_dict.get(field) for field in fields}
        list_of_comments.append(sub_dict)

    shuffle(list_of_comments)

    with open("Izlam_comments_training_data.txt", "w") as f:
        json.dump(list_of_comments[:500], f)

    with open("Izlam_comments_development_data.txt", "w") as f:
        json.dump(list_of_comments[500:750], f)

    with open("Izlam_comments_test_data.txt", "w") as f:
        json.dump(list_of_comments[750:1000], f)


def load_comments_for_dankchristianmemes():
    f = open("dcm_comments.txt", "r")
    ans = json.load(f)
    print(type(ans))

    for a in ans:
        print(a['body'])
#I will use dankchristianmemes and Izlam
if __name__ == "__main__":
    reddit = praw.Reddit(client_id='cT3V6CVDiqYhpw', client_secret='SccPhOMfwp-9cJMqrN44v6eR_oo', password='Charlie062799', 
        username = 'Lohikaarme27', user_agent = 'Test')
    reddit.config.store_json_result = True
    store_comments_for_dankchristianmemes(reddit)
    store_comments_for_izlam(reddit)
    #load_comments_for_dankchristianmemes()