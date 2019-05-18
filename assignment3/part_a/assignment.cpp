//
// Created by Dekai WU and YAN Yuchen on 20190417.
//

#include "assignment.hpp"
#include <shared_utils.hpp>

namespace part_a{
/*
  // ================== ITERATION 1 - HYPOTHESIS 1 =========================//
  // =======================================================================//

  const unsigned NUM_EPOCHS = 25;
  transducer_t your_classifier(const vector<sentence_t> &training_set, const vector<symbol_t> &postags,
                               const vector<symbol_t> &iobes_tags) {
    frequent_token_collector vocab_collector;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        vocab_collector.add_occurence(token);
      }
    }
    vector<symbol_t> vocab = vocab_collector.list_frequent_tokens(1000);
    auto embedding_lookup = make_embedding_lookup(64, vocab);
    auto pos_embed = make_embedding_lookup(64, postags);
    auto concatenate = make_concatenate(2);
    auto dense0 = make_dense_feedfwd(128, make_tanh());
    auto dense1 = make_dense_feedfwd(iobes_tags.size(), make_softmax());
    auto onehot_inverse = make_onehot_inverse(iobes_tags);
    return compose(group(embedding_lookup, pos_embed), concatenate, dense0, dense1, onehot_inverse);
  }

  vector<feature_t>
  get_features(const vector<symbol_t> &sentence, const vector<symbol_t> &postags, unsigned target_index) {

    return vector<feature_t>{sentence[target_index], postags[target_index]};
  }


  // ================= ITERATION 1 - HYPOTHESIS 2========================//
  // ===================================================================//

  const unsigned NUM_EPOCHS = 40;
  transducer_t your_classifier(const vector<sentence_t> &training_set, const vector<symbol_t> &postags,
                               const vector<symbol_t> &iobes_tags) {
    frequent_token_collector vocab_collector;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        vocab_collector.add_occurence(token);
      }
    }
    vector<symbol_t> vocab = vocab_collector.list_frequent_tokens(1000);
    auto embedding_lookup = make_embedding_lookup(256, vocab);
    auto pos_embed = make_embedding_lookup(256, postags);
    auto postag_onehot = make_onehot(postags);
    auto concatenate = make_concatenate(4);
    auto dense1 = make_dense_feedfwd(iobes_tags.size(), make_softmax());
    auto onehot_inverse = make_onehot_inverse(iobes_tags);
    return compose(group(embedding_lookup, pos_embed, pos_embed, pos_embed), concatenate, dense1, onehot_inverse);
  }

  vector<feature_t>
  get_features(const vector<symbol_t> &sentence, const vector<symbol_t> &postags, unsigned target_index) {
    feature_t prev;
    feature_t prev_tag;
    if (target_index > 0) {
	prev = sentence[target_index - 1];
	prev_tag = postags[target_index - 1];
     } else {
	prev = "<s>";
	prev_tag = "<s>";
    }
    
    return vector<feature_t>{sentence[target_index], prev, postags[target_index], prev_tag};
  }

  // ========== ITERATION 1 - HYPOTHESIS 3 ======================//
  // ===========================================================//
        
  const unsigned NUM_EPOCHS = 25;
  transducer_t your_classifier(const vector<sentence_t> &training_set, const vector<symbol_t> &postags,
                               const vector<symbol_t> &iobes_tags) {
    frequent_token_collector vocab_collector;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        vocab_collector.add_occurence(token);
      }
    }
    vector<symbol_t> vocab = vocab_collector.list_frequent_tokens(1000);
    auto embedding_lookup = make_embedding_lookup(64, vocab);
    auto pos_embed = make_embedding_lookup(64, postags);
    auto concatenate = make_concatenate(6);
    auto dense0 = make_dense_feedfwd(128, make_tanh());
    auto dense1 = make_dense_feedfwd(iobes_tags.size(), make_softmax());
    auto onehot_inverse = make_onehot_inverse(iobes_tags);

    return compose(group(embedding_lookup, pos_embed, pos_embed, pos_embed, pos_embed, pos_embed), concatenate, dense0, dense1, onehot_inverse);
  }

  vector<feature_t>
  get_features(const vector<symbol_t> &sentence, const vector<symbol_t> &postags, unsigned target_index) {

    feature_t prev;
    feature_t prev_tag;
    if (target_index > 0) {
	prev = sentence[target_index - 1];
	prev_tag = postags[target_index - 1];
     } else {
	prev = "<s>";
	prev_tag = "<s>";
    }
    
    feature_t next;
    feature_t next_tag;
    if (target_index < (sentence.size() - 1)) {
	next = sentence[target_index + 1];
	next_tag = postags[target_index + 1];
     } else {
	next = "<s>";
	next_tag = "<s>";
    }

    return vector<feature_t>{sentence[target_index], prev, next, postags[target_index], prev_tag, next_tag};

  }



  // ========== ITERATION 1 - HYPOTHESIS 4 ======================//
  // ===========================================================//
 
  const unsigned NUM_EPOCHS = 35;
  transducer_t your_classifier(const vector<sentence_t> &training_set, const vector<symbol_t> &postags, const vector<symbol_t> &iobes_tags) {
    frequent_token_collector vocab_collector;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        vocab_collector.add_occurence(token);
      }
    }
    vector<symbol_t> vocab = vocab_collector.list_frequent_tokens(1000);

    auto embedding_lookup = make_embedding_lookup(64, vocab);
    auto pos_embed = make_embedding_lookup(64, postags);
    auto concatenate = make_concatenate(4);
    auto dense0 = make_dense_feedfwd(128, make_tanh());
    auto dense1 = make_dense_feedfwd(iobes_tags.size(), make_softmax());
    auto onehot_inverse = make_onehot_inverse(iobes_tags);
    return compose(group(embedding_lookup, pos_embed, pos_embed, pos_embed), concatenate, dense0, dense1, onehot_inverse);
  }

  vector<feature_t>
  get_features(const vector<symbol_t> &sentence, const vector<symbol_t> &postags, unsigned target_index) {

    feature_t prev;
    feature_t prev_tag;
    if (target_index > 0) {
	//prev = sentence[target_index - 1];
	prev_tag = postags[target_index - 1];
     } else {
	//prev = "<s>";
	prev_tag = "<s>";
    }
    
    feature_t next;
    feature_t next_tag;
    if (target_index < (sentence.size() - 1)) {
	//next = sentence[target_index + 1];
	next_tag = postags[target_index + 1];
     } else {
	//next = "<s>";
	next_tag = "<s>";
    }
    return vector<feature_t>{sentence[target_index], prev_tag, postags[target_index], next_tag};
  }

  // ================== ITERATION 2 - HYPOTHESIS 1 =========================//
  // =======================================================================//

  const unsigned NUM_EPOCHS = 30;
  transducer_t your_classifier(const vector<sentence_t> &training_set, const vector<symbol_t> &postags,
                               const vector<symbol_t> &iobes_tags) {
    frequent_token_collector vocab_collector;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        vocab_collector.add_occurence(token);
      }
    }
    vector<symbol_t> vocab = vocab_collector.list_frequent_tokens(1000);
    auto embedding_lookup = make_embedding_lookup(128, vocab);
    auto pos_embed = make_onehot(postags);
    auto concatenate = make_concatenate(2);
    auto dense0 = make_dense_feedfwd(128, make_tanh());
    auto dense1 = make_dense_feedfwd(iobes_tags.size(), make_softmax());
    auto onehot_inverse = make_onehot_inverse(iobes_tags);
    return compose(group(embedding_lookup, pos_embed), concatenate, dense1, onehot_inverse);
  }

  vector<feature_t>
  get_features(const vector<symbol_t> &sentence, const vector<symbol_t> &postags, unsigned target_index) {

    return vector<feature_t>{sentence[target_index], postags[target_index]};
  }

*/
  // ================== ITERATION 2 - HYPOTHESIS 2 =========================//
  // =======================================================================//

  const unsigned NUM_EPOCHS = 30;
  transducer_t your_classifier(const vector<sentence_t> &training_set, const vector<symbol_t> &postags,
                               const vector<symbol_t> &iobes_tags) {
    frequent_token_collector vocab_collector;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        vocab_collector.add_occurence(token);
      }
    }
    vector<symbol_t> vocab = vocab_collector.list_frequent_tokens(1000);
    auto embedding_lookup = make_embedding_lookup(64, vocab);
    auto pos_embed = make_onehot(postags);
    auto concatenate = make_concatenate(8);
    auto dense0 = make_dense_feedfwd(64, make_tanh());
    auto dense1 = make_dense_feedfwd(iobes_tags.size(), make_softmax());
    auto onehot_inverse = make_onehot_inverse(iobes_tags);
    return compose(group(embedding_lookup, embedding_lookup, embedding_lookup, pos_embed, pos_embed, pos_embed, pos_embed, pos_embed), concatenate, dense0, dense1, onehot_inverse);
  }

  vector<feature_t>
  get_features(const vector<symbol_t> &sentence, const vector<symbol_t> &postags, unsigned target_index) {
	feature_t prev = (target_index > 0) ? postags[target_index-1] : "<s>";
	feature_t prev_t = (target_index > 0) ? sentence[target_index-1] : "<s>";
	feature_t next_t = (int(target_index) < int(sentence.size())-1) ? sentence[target_index+1] : "<s>";
	feature_t next = (int(target_index) < int(postags.size())-1) ? postags[target_index+1] : "<s>";
 	feature_t prev1 = (target_index > 1) ? postags[target_index-2] : "<s>";
	feature_t next1 = (int(target_index) < int(postags.size())-2) ? postags[target_index+2] : "<s>";
    return vector<feature_t>{prev_t, sentence[target_index], next_t, prev1, prev, postags[target_index], next, next1};
  }

}
