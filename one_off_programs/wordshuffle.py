import lorem
import numpy as np 


data = lorem.paragraph()
words = data.split()

# try 1
# need to handle punctuation
# index punctuation
#shuffled = [word.replace(word[1:-1],''.join(np.random.permutation(list(word[1:-1])))) for word in words if len(word)>3 else ]

# try 2
# handle punctuation at end of words
shuffle = words[:]
for word in shuffle:
    if word.isalnum(): #no punctuation
        if len(word)>3:
            print('shuffle: {:s}'.format(word))
            word = list(word)
            word[1:-1] = np.random.permutation(list(word[1:-1]))
            word = ''.join(word)
            print('shuffle: {:s}'.format(word))
    else:
        if word[:-1].isalnum(): #if there is punctuation
            if len(word[:-1])>3: #onle shuffle upto next to last character
                print('shuffle: {:s}'.format(word))
                word = list(word)
                word[1:-2] = np.random.permutation(list(word[1:-2]))
                word = ''.join(word)
                print('shuffle: {:s}'.format(word))

    
#try 3. trying to be clever
# turned out to be very pythonic 
# by using a function
# also makes it easily modifiable
data = 'Labore eius velit adipisci.'
data = lorem.paragraph()
words = data.split()

def dothis(w):
    """shuffle middle characters of a word
    Arguments:
        w {stirng} -- word you want middle shuffled
    Returns:
        string -- shuffled word
    """
    # .isalnum tests for if it has punctuation. I assume it is at end of word
    s = w[0] #get first character
    m = list(w[1:-1] if w.isalnum() else w[1:-2]) #get middle characters or empty
    e = w[-1] if w.isalnum() else w[-2:] #get last characters

    # use while loop to prevent shuffling
    # to the same value
    # common if shuffling 2 letters
    while (m==list(w[1:-1] if w.isalnum() else w[1:-2])) and (len(m)>1):
        np.random.shuffle(m)

    # concat as lists for joining later
    h = [s] + m + [e] #can only join arrays

    print('shuffled {:s} to {:s}'.format(w,''.join(h)))
    return ''.join(h)

words = [dothis(w) for w in words if len(w)>3]
paragraph = ' '.join(words)