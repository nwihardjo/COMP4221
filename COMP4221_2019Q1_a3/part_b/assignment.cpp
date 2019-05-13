//
// Created by Dekai WU and YAN Yuchen on 20190417.
//

#include "assignment.hpp"
#include <shared_utils.hpp>

namespace part_b{
  const unsigned NUM_EPOCHS = 5;

  /**
   * create your custom classifier by combining transducers
   * the input to your classifier will a list of tokens
   * when creating your custom classifier, the training set is passed as a parameter
   * this is because you need to assemble your vocabulary from training set
   * \param training_set the training set that your classifier will train on
   * \param postags the list of all POS tags.
   * \param iobes_tags the list of IOBES tags. it contains "I" "O" "B" "E" "S" (but not necessarily in order)
   * \return
   */
  transducer_t your_classifier(const vector<sentence_t> &training_set, const vector<symbol_t> &postags,
                               const vector<symbol_t> &iobes_tags) {

    // in this starting code, we demonstrates how to construct a 2-layer feedforward neural network
    // that takes the target token and the POS tag of the target token as input

    // first you need to assemble the vocab you need
    // in this simple model, the vocab is the top 1000 most frequent tokens in training set
    // we provide a frequent_token_collector utility,
    // that can count token frequencies and collect the top X most frequent tokens
    // all out-of-vocabulary tokens will be treated as "unknown token"
    frequent_token_collector vocab_collector;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        vocab_collector.add_occurence(token);
      }
    }
    vector<symbol_t> vocab = vocab_collector.list_frequent_tokens(1000);

    // create an embedding lookup layer for token input
    auto embedding_lookup = make_embedding_lookup(64, vocab);

    // the size of postag vocabulary are small, a onehot layer will work just fine
    auto postag_onehot = make_onehot(postags);

    auto concatenate = make_concatenate(4);

    auto dense0 = make_dense_feedfwd(64, make_tanh());

    auto dense1 = make_dense_feedfwd(iobes_tags.size(), make_softmax());

    auto onehot_inverse = make_onehot_inverse(iobes_tags);

    return compose(group(embedding_lookup, embedding_lookup,embedding_lookup,embedding_lookup), concatenate, dense0, dense1, onehot_inverse);
  }

  /**
   * besides the target token to chunk, your model may also need other "context" input
   * this function defines the inputs that your model expects
   * \param sentence the sentence where the token is in
   * \param postags the POS tags of the sentence (predicted by your model)
   * \param target_index the position of the target token to chunk
   * \return
   */
  vector<feature_t>
  get_features(const vector<symbol_t> &sentence, const vector<symbol_t> &postags, unsigned target_index) {

    // TODO: define what input to feed to your classifier
    // this starting code demonstrates how to define input as:
    // the target token, the target token's POS tag
    if(target_index>0){
      return vector<feature_t>{sentence[target_index], postags[target_index],postags[target_index-1],sentence[target_index-1]};}
    else {return vector<feature_t>{sentence[target_index], postags[target_index],postags[target_index],sentence[target_index]};}
  }
}
