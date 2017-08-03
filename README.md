# Tensorflow examples for the longmont data science meetup
Contained in this repository are a data file and an example in tensorflow that puts this data to use. The data is a lot of text, all shakespeare, chosen for its recognizability and relative lack of controversy. The use to which it is put is contained in `Tell_me_a_story.py` which will learn the shakespearean text, and if called as `./Tell_me_a_story.py --mode sample` will attempt to generate some new shakesperean text from scratch, using the tensorflow model you've constructed.

There are two subdirectories included, one called `logs` which contains outputs for tensorboard information. If you wish, try `tensorboard --logdir logs/` from the base directory of the project while training and your network and traning information will be available.

The other subdirectory is called `save` which will contain the checkpoints your model is generating for use later.

## tensorflow example
This example has relatively few bells and whsitles, opting instead to be extremely compact, complete, and a solid base for your own personal experimentation. It mirrors a subset of the performance of this [char-rnn-tensorflow model](https://github.com/sherjilozair/char-rnn-tensorflow) without either being an exact copy of the functionality or the construction.

This example was chosen because, broadly, everyone is familiar with text and what it should look like, so rather than attempting some state-of-the-art performance metric on a task that is not readily explanable, the ability of the model to generate understandable text is a gentler goal for the learner.


## Interesting things to try.
1. Modify the text parser to include different punctuation and see how long it takes to use it in a reasonable way.
2. Change out the training set for your favorite text and see what it comes up with. I tried this with my own journals at one point. It really did capture my tone, but in a way that was more eerie than fun.
3. Implement a different RNN type, try using GRU.
3. Add dropout to prevent overfitting to data
4. Use more data and see the effect of the variety on the accuracy and output.
4. Implement and see the effect of gradient clipping.
5. Change the number of layers or the number of nodes and see the effect on training.
6. Modify the embedding layer to layers, or by moving the number of nodes around.
7. Add more post-processing layers.
8. Each choice made in this example is endorsed or encouraged by some set of arxiv papers and likely by another set. Try to find some of each and understand the rationales behind those assertions.
