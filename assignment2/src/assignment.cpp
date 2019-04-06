#include "assignment.hpp"
using namespace tg;

// TODO: fill in your student ID
const char *STUDENT_ID = "00000000";

// TODO: choose number of epochs to train
const unsigned NUM_EPOCHS = 5;

/**
 * create your custom classifier by combining transducers
 * the input to your classifier will a list of tokens
 * \param vocab a list of all possible words
 * \param postags a list of all possible POS tags
 * \return
 */
transducer_t your_classifier(const vector<token_t> &vocab, const vector<postag_t> &postags) {

  // in this starting code, we demonstrates how to construct a 2-layer feedforward neural network
  // that takes 2 tokens as features

  // embedding lookup layer is mathematically equivalent to
  // a 1-hot layer followed by a dense layer with identity activation
  // but trains faster then composing those two separately
  auto embedding_lookup = make_embedding_lookup(64, vocab);

  // create a concatenation operation
  // this operation can concatenate the 2 tokens (in 1-hot representation) into a big vector feature
  auto concatenate = make_concatenate(2);

  // create the first feedforward layer,
  // with 64 output units and tanh as activation function
  auto dense0 = make_dense_feedfwd(64, make_tanh());

  // create another feedforward layer
  // this is the final layer. so this layer should return the predicted POS tag (in 1-hot representation)
  // the output dimension of your final layer should be the size of all POS tags
  // because each dimension corresponds to a particular choice
  auto dense1 = make_dense_feedfwd((unsigned) postags.size(), make_softmax());

  // this is the inverse 1-hot operation
  // it takes a 1-hot vector feature, and returns a token from a pre-defined vocabulary
  // the 1-hot vector feature doesn't have to be perfect 0 and 1 values.
  // but they have to sum up to 1 (just like probability distribution)
  // usually the this (approximated) 1-hot input comes from a softmax operation
  auto onehot_inverse = make_onehot_inverse(postags);

  // connect these layers together
  // composing A and B means first apply A, and then take the output of A and feed into B
  return compose(group(embedding_lookup, embedding_lookup), concatenate, dense1, onehot_inverse);
}

// rename me into your_classifier if you want to use this classifier
transducer_t your_classifier_knn(const vector<token_t> &vocab, const vector<postag_t> &postags) {

  // in this starting code, we demonstrates how to construct a KNN that takes two tokens as features

  // a KNN classifier takes a real-valued vector feature and directly returns the predicted class
  auto knn = make_symbolic_k_nearest_neighbors_classifier(5, 2, postags);

  return knn;
}


/**
 * besides the target token to classify, your model may also need other tokens as "context" input
 * this function defines the inputs that your model expects
 * \param sentence the sentence that the target token is coming from
 * \param token_index the position of the target token in the sentence
 * \return a list of tokens to feed to your model
*/
vector<token_t> get_features(const vector<token_t> &sentence, unsigned token_index) {

  // in these starting code, we demonstrate how to feed the target token,
  // together with its preceding token as context.

  if (token_index > 0) {

    // when the target token is not the first token of the sentence, we just need to
    // feed the target token and its previous token to the model
    return vector<token_t>{sentence[token_index], sentence[token_index - 1]};
  } else {

    // in case the target token is the first token, we need to invent a dummy previous token.
    // this is because a feedforward neural network expects consistent input dimensions.
    // if sometimes you give the feedforward neural network 1 token as input,
    // sometimes you give it 2 tokens as input, then the feedforward neural network will be angry.

    // there is nothing special about the string "<s>". you can pick whatever you want as long as
    // it doesn't appear in the vocabulary
    return vector<token_t>{sentence[token_index], "<s>"};
  }
}
