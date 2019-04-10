#include "assignment.hpp"
#include<vector>
#include<iostream>
using namespace tg;

const char *STUDENT_ID = "20315011";

void lower_case(token_t &str){
	//transform(str.begin(), str.end(), str.begin(), ::tolower);
}

/* 
// ITERATION 1 - HYPOTHESIS 1 
const unsigned NUM_EPOCHS = 40; 
transducer_t your_classifier(const vector<token_t> &vocab, const vector<postag_t> &postags) { auto embedding_lookup = make_embedding_lookup(64, vocab);
  auto concatenate = make_concatenate(2);
  auto dense0 = make_dense_feedfwd(64, make_tanh());
  auto dense1 = make_dense_feedfwd((unsigned) postags.size(), make_softmax());
  auto onehot_inverse = make_onehot_inverse(postags);
  return compose(group(embedding_lookup, embedding_lookup), concatenate, dense0, dense1, onehot_inverse);
}
vector<token_t> get_features(const vector<token_t> &sentence, unsigned token_index) {
  if (token_index > 0) {
    return vector<token_t>{sentence[token_index], sentence[token_index - 1]};
  } else {
    return vector<token_t>{sentence[token_index], "<s>"};
  }
}

// ITERATION 1 - HYPOTHESIS 2
const unsigned NUM_EPOCHS = 30;
transducer_t your_classifier(const vector<token_t> &vocab, const vector<postag_t> &postags) { 
  auto embedding_lookup = make_embedding_lookup(64, vocab);
  auto concatenate = make_concatenate(5);
  auto dense0 = make_dense_feedfwd(64, make_tanh());
  auto dense1 = make_dense_feedfwd((unsigned) postags.size(), make_softmax());
  auto onehot_inverse = make_onehot_inverse(postags);
  return compose(group(embedding_lookup, embedding_lookup, embedding_lookup, embedding_lookup, embedding_lookup), concatenate, dense0, dense1, onehot_inverse);
}
vector<token_t> get_features(const vector<token_t> &sentence, unsigned token_index) {
  if (token_index > 4) {
    return vector<token_t>{sentence[token_index], sentence[token_index - 1], sentence[token_index - 2], sentence[token_index-3], sentence[token_index-4]};
  } else if (token_index == 3) {
    return vector<token_t>{sentence[token_index], sentence[token_index - 1], sentence[token_index-2], sentence[token_index-3], "<s>"};
  } else if (token_index == 2) {
    return vector<token_t>{sentence[token_index], sentence[token_index - 1], sentence[token_index-2], "<s>", "<s>"};
  } else if (token_index == 1) {
    return vector<token_t>{sentence[token_index], sentence[token_index - 1], "<s>", "<s>", "<s>"};
  } else {
    return vector<token_t>{sentence[token_index], "<s>", "<s>", "<s>", "<s>"};
  }
}

// ITERATION 1 - HYPOTHESIS 3
const unsigned NUM_EPOCHS = 10;
transducer_t your_classifier(const vector<token_t> &vocab, const vector<postag_t> &postags) { 
  auto embedding_lookup = make_embedding_lookup(256, vocab);
  auto concatenate = make_concatenate(3);
  auto dense0 = make_dense_feedfwd(256, make_tanh());
  auto dense_ = make_dense_feedfwd(64, make_tanh());
  auto dense1 = make_dense_feedfwd((unsigned) postags.size(), make_softmax());
  auto onehot_inverse = make_onehot_inverse(postags);
  return compose(group(embedding_lookup, embedding_lookup, embedding_lookup), concatenate, dense0, dense1, onehot_inverse);
}
*/
vector<token_t> get_features(const vector<token_t> &sentence, unsigned token_index) {
  auto size = sentence.size();
  token_t previous;
  token_t next;
  token_t current = sentence[token_index];

  if (token_index > 0 ){
    previous = sentence[token_index - 1];
    lower_case(previous);
  }
  else
    previous = "<s>";

  if (token_index < size - 1){
    next = sentence[token_index + 1];
    lower_case(next);
  }
  else
    next = "<s>";

  lower_case(current);
  return vector<token_t>{previous, current, next};
}
/*
// ITERATION 2 - HYPOTHESIS 2
const unsigned NUM_EPOCHS = 50;
transducer_t your_classifier(const vector<token_t> &vocab, const vector<postag_t> &postags) { 
  auto embedding_lookup = make_embedding_lookup(64, vocab);
  auto concatenate = make_concatenate(5);
  auto dense0 = make_dense_feedfwd(64, make_tanh());
  auto dense_ = make_dense_feedfwd(64, make_tanh());
  auto dense1 = make_dense_feedfwd((unsigned) postags.size(), make_softmax());
  auto onehot_inverse = make_onehot_inverse(postags);
  return compose(group(embedding_lookup, embedding_lookup, embedding_lookup, embedding_lookup, embedding_lookup), concatenate, dense0, dense_, dense1, onehot_inverse);
}
vector<token_t> get_features(const vector<token_t> &sentence, unsigned token_index) {
  int size = sentence.size();
  token_t current = sentence[token_index]; 
  lower_case(current);
  token_t temp;
  token_t temp_;
  token_t temp1;
  token_t temp1_;

  if (token_index > 1) {
    temp = sentence[token_index - 2]; lower_case(temp);
    temp_ = sentence[token_index - 1]; lower_case(temp_);
  } else if (token_index == 1) {
    temp = "<s>";
    temp_ = sentence[token_index - 1]; lower_case(temp);
  } else {
    temp = "<s>";
    temp_ = "<s>";
  }

  if (int(token_index) < (size - 2) ){
    temp1 = sentence[token_index + 2]; lower_case(temp);
    temp1_ = sentence[token_index + 1]; lower_case(temp_);
  } else if (int(token_index) == (size - 2)) {
    temp1 = "<s>";
    temp1_ = sentence[token_index + 1]; lower_case(temp);
  } else {
    temp1 = "<s>";
    temp1_ = "<s>";
  }

  return vector<token_t>{temp, temp_, current, temp1_, temp1};
}
*/

// ITERATION 1 - HYPOTHESIS 4
const unsigned NUM_EPOCHS = 60;
transducer_t your_classifier(const vector<token_t> &vocab, const vector<postag_t> &postags) { 
  auto embedding_lookup = make_embedding_lookup(128, vocab);
  auto concatenate = make_concatenate(5);
  auto dense0 = make_dense_feedfwd(64, make_tanh());
  auto dense_ = make_dense_feedfwd(32, make_tanh());
  auto dense1 = make_dense_feedfwd((unsigned) postags.size(), make_softmax());
  auto onehot_inverse = make_onehot_inverse(postags);
  return compose(group(embedding_lookup, embedding_lookup, embedding_lookup, embedding_lookup), concatenate, dense0, dense1, onehot_inverse);
}
vector<token_t> get_features(const vector<token_t> &sentence, unsigned token_index) {
  token_t suffix;
  token_t first_letter;
  
  bool isCapital = isupper(sentence[token_index].at(0);
  if isCapital
    first_letter = "1";
  else
    first_letter = "0";

  if (token_t.length() > 3  
    suffix = sentence[token_index].substr(sentence[token_index].length() - 3);
  else
    suffix = "<s>";

  token_t previous
  token_t next;
  token_t current = sentence[token_index];

  if (token_index > 0 ){
    previous = sentence[token_index - 1];
    lower_case(previous);
  }
  else
    previous = "<s>";

  if (token_index < size - 1){
    next = sentence[token_index + 1];
    lower_case(next);
  }
  else
    next = "<s>";

  lower_case(current);
  return vector<token_t>{previous, current, next, first_letter, suffix};
}

/*
// rename me into your_classifier if you want to use this classifier
const unsigned NUM_EPOCHS =1000;
transducer_t your_classifier (const vector<token_t> &vocab, const vector<postag_t> &postags) {

  // in this starting code, we demonstrates how to construct a KNN that takes two tokens as features

  // a KNN classifier takes a real-valued vector feature and directly returns the predicted class
  auto knn = make_symbolic_k_nearest_neighbors_classifier(1, 3, postags);

  return knn;
}
*/
