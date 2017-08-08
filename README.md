# Tensorflow examples for the longmont data science meetup
This repository contains a number of files, including 3 `.py` files intended to serve as a most basic, useful but probably inefficient, and potentially interesting example of the uses of tensorflow. These are `basics.py`, `minima.py` and `Tell_me_a_story.py` respectively. `basics.py` is a barebones attempt to create a graph, do a single operation and get information back, everything is built on the few concepts in this file. `minima.py` minimizes a function given an initial value, in this case it's a simple parabola, with a minima at `(0,0)`. Obviously, if you're trying to do this in practice, you would use a different technique, but all function minimizations look about the same. Try modifying this one to minimize a function of your choice.


# Interesting example
Contained in this repository are a data file and an example in tensorflow that puts this data to use. The data is a lot of text, all shakespeare, chosen for its recognizability and relative lack of controversy. The use to which it is put is contained in `Tell_me_a_story.py` which will learn the shakespearean text as well as outputting a set of predictions and generated text every so often. The generated text can, if the model is run big enough and long enough, reach a kind of understandability, but the model will need modification. Consider the `char-rnn-tensorflow` model referenced below to get this level of performance without any additional work.

There is a subdirectory included, called `logs` which contains outputs for tensorboard information. If you wish, try `tensorboard --logdir logs/` from the base directory of the project while training and your network and traning information will be available.

## tensorflow example
This example has relatively few bells and whsitles, opting instead to be extremely compact, complete, and a solid base for your own personal experimentation. It mirrors a subset of the performance of this [char-rnn-tensorflow model](https://github.com/sherjilozair/char-rnn-tensorflow) without either being an exact copy of the functionality or the construction.

This example was chosen because, broadly, everyone is familiar with text and what it should look like, so rather than attempting some state-of-the-art performance metric on a task that is not readily explanable, the ability of the model to generate understandable text is a gentler goal for the learner.


## Interesting things to try.
1. Modify the text parser to include different punctuation and see how long it takes to use it in a reasonable way.
2. Change out the training set for your favorite text and see what it comes up with. I tried this with my own journals at one point. It really did capture my tone, but in a way that was more eerie than fun.
3. Implement a different RNN type, try using LSTM.
3. Add dropout to the dense nodes.
4. Implement saving and restoring of the model.
4. Use more data and see the effect of the variety on the accuracy and output.
4. Train in epochs with decaying learning rate.
4. Implement and see the effect of gradient clipping.
5. Change the number of layers or the number of nodes and see the effect on training.
6. Modify the embedding layer to layers, or by moving the number of nodes around.
7. Add more post-processing layers.
8. Each choice made in this example is endorsed or encouraged by some set of arxiv papers and likely by another set. Try to find some of each and understand the rationales behind those assertions.

If you implement a few of the above and run the model for a long while, you can get the autonomous predictions looking something like this below:


> vio cousin buckingham with a father in his son to the boy. the prince good death now what they shall not make us do in a true eyes to find content to do so to him for what they shall not have a proper enemies know not your father is to 't. the consent of death and good souls where you are in my country's anger the true consent so sorrow of all the dead many soundly be fair to long consent to your worship of his name with him for my son reason in consent of my life in death of your worship in your son, and i know what look the truth hath for some soul but my brother and the rest have been many words like part of us but one for the consent not a father will not be so to the sun of his good prayers to my man unto the house of york 'tis in the consent of prove it with good and entreation and many that she may not the rest your manners be his death to find a true the good and fair off the prince the rest and good for ever are done with a word with your father shall rest and here for my country's brother lies have comes upon my head of english be burnt of some death with him and the consent of my good to see the 'tis the heavy death upon the better your love down so fair consul.  romeo: i cannot be a word by in presence and here in the man in his son of death hath he do the world upon the son, that love the instruments your honour with the death by what he comes to his consent to prisoner man your presence and prove the joy proud prove a proper that have you to vengeance of good delight prove which a consul with me of my father that i go on the english his head upon the world in the english be prove a word of the world by who do you not so great lord prayers and death and good and so much unto a father and a bank upon my man that do it not so longer to see the consent to prisoner to be contrary in a father may do him and your father's son of your son, i repent the man unto romeo go and bear the land upon our death by fortune and many liege, but some that work to me of the
