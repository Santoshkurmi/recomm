import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

type_weights = {
    'liked':2,
    'created':3,
    'viewed':1,
}

# posts = created[created
#     {'title':'How to fix this php code', 'description':'How to make it sure to have going in those problem.','type':'liked'},
#     {'title':'Java code is not running', 'description':'How to print hello world in java.','type':'created'},
#     {'title':'React router is showing error', 'description':'React router throughing error about something','type':'viewed'},
#     {'title':'React dom not working for me.', 'description':'React router is not working properly','type':'liked'},
#     {'title':'How to create a box in html', 'description':'I am trying to make square box in html but its not working at all','type':'created'},
#     {'title':'Make this php code fix please', 'description':'This php code is throwing error even everything is correct','type':'viewed'},
#     {'title':'event listener in html', 'description':'This is just not working at all','type':'created'},
# ]


# vectorizer = TfidfVectorizer(stop_words='english')
vectorizer = TfidfVectorizer()


def calculateNewProfile(postData,weight):

    df = pd.DataFrame(postData)

    df['text'] = df['title'] + ' ' + df['description']

    df['weight'] = df['type'].map(weight)

    # print(df['weight'])

    tfidf_data =  vectorizer.transform(df['text'])

    weightted_tfidf =  tfidf_data.multiply(df['weight'].values[:,None])
    # print(weightted_tfidf)

    good_view = pd.DataFrame(weightted_tfidf.toarray(),columns=vectorizer.get_feature_names_out())
    print(good_view)
    return weightted_tfidf
    # user_profile_vector = np.mean(weightted_tfidf,axis=0)

    # print(user_profile_vector)

    # user_profile_array = np.asarray(user_profile_vector)

    return weightted_tfidf





def calculateProfile(postData,weight):

    df = pd.DataFrame(postData)

    df['text'] = df['title'] + ' ' + df['description']

    df['weight'] = df['type'].map(weight)

    # print(df['weight'])

    tfidf_data =  vectorizer.fit_transform(df['text'])
    weightted_tfidf =  tfidf_data.multiply(df['weight'].values[:,None])

    better_view = pd.DataFrame(weightted_tfidf.toarray(),columns=vectorizer.get_feature_names_out() )
    print(better_view)
    
    return tfidf_data



    # user_profile_vector = np.mean(weightted_tfidf,axis=0)

    # # print(user_profile_vector)

    # user_profile_array = np.asarray(user_profile_vector)

    # return weightted_tfidf

posts = [
    {'title':'apple', 'description':'ball ','type':'liked'},
    {'title':'ball', 'description':'cat','type':'created'},
    {'title':'cat', 'description':'dog','type':'viewed'},
]

newPost = [
    {'title':'ball', 'description':'cat is awesome in this way','type':'liked'},
    {'title':'apple', 'description':'cat is awesome in this way','type':'liked'},
    {'title':'apple', 'description':'ball cat is awesome in this way','type':'liked'},
]

user_profile = calculateProfile(posts,type_weights)
new_profile = calculateNewProfile(newPost,type_weights)

similarity = cosine_similarity(new_profile,user_profile)
np_smilarity = np.array(similarity)

summ = np.sum(np_smilarity,axis=1)
# sorted = np.sort(summ)[::-1]

print(similarity,end='\n\n')
print(summ)

# print(user_profile)



# post_to_choose_from = [
#     {'title':'How to fix this php code', 'description':'How to make it sure to have going in those problem.','type':'liked'},
#     {'title':'Java code is not running', 'description':'How to print hello world in java.','type':'created'},

# ]

# profile_new = calculateNewProfile(post_to_choose_from,type_weights)


# similarity = cosine_similarity(user_profile,profile_new)

# # user_profile_vector = np.mean(similarity,axis=0)


# print( similarity )

# print(profile_new)

# print(user_profile)