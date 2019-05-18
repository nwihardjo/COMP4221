//
// Created by Dekai WU and YAN Yuchen on 20190417.
//

#include "assignment.hpp"
#include <shared_utils.hpp>

namespace part_b{
/*
  // ==========================
  // ITERATION 1 - HYPOTHESIS 1
  // ==========================

  const unsigned NUM_EPOCHS = 20;

  transducer_t your_classifier(const vector<sentence_t> &training_set, const vector<symbol_t> &postags, const vector<symbol_t> &iobes_tags) {

    frequent_token_collector vocab_collector;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        vocab_collector.add_occurence(token);
      }
    }
    vector<symbol_t> vocab = vocab_collector.list_frequent_tokens(1000);

    auto embedding_lookup = make_embedding_lookup(64, postags);

    auto concatenate = make_concatenate(2);

    auto dense0 = make_dense_feedfwd(64, make_tanh());

    auto dense1 = make_dense_feedfwd(iobes_tags.size(), make_softmax());

    auto onehot_inverse = make_onehot_inverse(iobes_tags);

	return compose(embedding_lookup, dense1, onehot_inverse);
  }

  vector<feature_t>
  get_features(const vector<symbol_t> &sentence, const vector<symbol_t> &postags, unsigned target_index) {
	return vector<feature_t>{postags[target_index]};
  }

  // ==========================
  // ITERATION 1 - HYPOTHESIS 2
  // ==========================

  const unsigned NUM_EPOCHS = 40;

  transducer_t your_classifier(const vector<sentence_t> &training_set, const vector<symbol_t> &postags, const vector<symbol_t> &iobes_tags) {

    frequent_token_collector vocab_collector;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        vocab_collector.add_occurence(token);
      }
    }
    vector<symbol_t> vocab = vocab_collector.list_frequent_tokens(1000);

    auto embedding_lookup = make_embedding_lookup(64, postags);

    auto concatenate = make_concatenate(3);

    auto dense0 = make_dense_feedfwd(64, make_tanh());

    auto dense1 = make_dense_feedfwd(iobes_tags.size(), make_softmax());

    auto onehot_inverse = make_onehot_inverse(iobes_tags);

    return compose(group(embedding_lookup, embedding_lookup, embedding_lookup), concatenate, dense1, onehot_inverse);
  }

  vector<feature_t> get_features(const vector<symbol_t> &sentence, const vector<symbol_t> &postags, unsigned target_index) {
    
    feature_t prev = (target_index > 0) ? postags[target_index-1] : "<s>";
    feature_t next = (target_index < sentence.size() - 1) ? postags[target_index+1] : "<s>";

    return vector<feature_t>{prev, postags[target_index], next};
  }


  // ==========================
  // ITERATION 1 - HYPOTHESIS 3
  // ==========================

  const unsigned NUM_EPOCHS = 50;

  transducer_t your_classifier(const vector<sentence_t> &training_set, const vector<symbol_t> &postags, const vector<symbol_t> &iobes_tags) {

    frequent_token_collector vocab_collector;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        vocab_collector.add_occurence(token);
      }
    }
    vector<symbol_t> vocab = vocab_collector.list_frequent_tokens(1000);

    auto embedding_lookup = make_embedding_lookup(128, vocab);

    auto postag = make_embedding_lookup(128, postags);

    auto concatenate = make_concatenate(2);

    auto dense0 = make_dense_feedfwd(64, make_tanh());

    auto dense1 = make_dense_feedfwd(iobes_tags.size(), make_softmax());

    auto onehot_inverse = make_onehot_inverse(iobes_tags);

    return compose(group(embedding_lookup, postag), concatenate, dense1, onehot_inverse);
  }

  vector<feature_t> get_features(const vector<symbol_t> &sentence, const vector<symbol_t> &postags, unsigned target_index) {
    
    return vector<feature_t>{sentence[target_index], postags[target_index]};
  }


  // ==========================
  // ITERATION 1 - HYPOTHESIS 4 
  // ==========================
  const unsigned NUM_EPOCHS = 60;

  transducer_t your_classifier(const vector<sentence_t> &training_set, const vector<symbol_t> &postags, const vector<symbol_t> &iobes_tags) {
    frequent_token_collector vocab_collector;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        vocab_collector.add_occurence(token);
      }
    }
    vector<symbol_t> vocab = vocab_collector.list_frequent_tokens(1000);

    auto embedding_lookup = make_embedding_lookup(64, vocab);
    auto postag = make_embedding_lookup(32, postags);
    auto concatenate = make_concatenate(6);
    auto dense0 = make_dense_feedfwd(64, make_tanh());
    auto dense1 = make_dense_feedfwd(iobes_tags.size(), make_softmax());
    auto onehot_inverse = make_onehot_inverse(iobes_tags);

    return compose(group(embedding_lookup, embedding_lookup, embedding_lookup, postag, postag, postag), concatenate, dense1, onehot_inverse);
  }

  vector<feature_t> get_features(const vector<symbol_t> &sentence, const vector<symbol_t> &postags, unsigned target_index) {
    feature_t next;
    feature_t next_pos;
    feature_t prev;
    feature_t prev_pos;

    if (target_index > 0) {
	prev = sentence[target_index-1];
	prev_pos = postags[target_index-1];
    } else {
	prev = "<s>";
	prev_pos = "<s>";
    }

    if (int(target_index) < int(sentence.size())-1) {
	next = sentence[target_index+1];
	next_pos = postags[target_index+1];
    } else {
	next = "<s>";
	next_pos = "<s>";
    }
    return vector<feature_t>{prev, sentence[target_index], next, prev_pos, postags[target_index], next_pos};
  }

  // ==========================
  // ITERATION 2 - HYPOTHESIS 1
  // ==========================

  const unsigned NUM_EPOCHS = 40;

  transducer_t your_classifier(const vector<sentence_t> &training_set, const vector<symbol_t> &postags, const vector<symbol_t> &iobes_tags) {

    frequent_token_collector vocab_collector;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        vocab_collector.add_occurence(token);
      }
    }
    vector<symbol_t> vocab = vocab_collector.list_frequent_tokens(1000);

    auto postag_onehot = make_onehot(postags);

    auto concatenate = make_concatenate(2);

    auto dense0 = make_dense_feedfwd(64, make_tanh());

    auto dense1 = make_dense_feedfwd(iobes_tags.size(), make_softmax());

    auto onehot_inverse = make_onehot_inverse(iobes_tags);

	return compose(postag_onehot, dense0, dense1, onehot_inverse);
  }

  vector<feature_t>
  get_features(const vector<symbol_t> &sentence, const vector<symbol_t> &postags, unsigned target_index) {
	return vector<feature_t>{postags[target_index]};
  }

  // ==========================
  // ITERATION 2 - HYPOTHESIS 2 
  // ==========================
  const unsigned NUM_EPOCHS = 50;

  transducer_t your_classifier(const vector<sentence_t> &training_set, const vector<symbol_t> &postags, const vector<symbol_t> &iobes_tags) {

    frequent_token_collector vocab_collector;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        vocab_collector.add_occurence(token);
      }
    }
    vector<symbol_t> vocab = vocab_collector.list_frequent_tokens(1000);

    auto embedding_lookup = make_embedding_lookup(64, vocab);
    auto pos_embed = make_embedding_lookup(64, vocab);
    auto postag = make_onehot(postags);

    auto concatenate = make_concatenate(8);

    auto dense1 = make_dense_feedfwd(iobes_tags.size(), make_softmax());

    auto onehot_inverse = make_onehot_inverse(iobes_tags);

    return compose(group(embedding_lookup, postag, postag, postag, postag, postag, postag, postag), concatenate, dense1, onehot_inverse);
  }

  vector<feature_t> get_features(const vector<symbol_t> &sentence, const vector<symbol_t> &postags, unsigned target_index) {
    feature_t next_pos;
    feature_t prev_pos;
    feature_t next = (target_index > 3) && (int(postags.size()) > 3) ? postags[target_index-3] : "<s>";
    feature_t prev = (int(target_index) < int(postags.size())-3) ? postags[target_index+3] : "<s>";

    feature_t prev1 = (target_index > 2) ? postags[target_index-2] : "<s>";
    feature_t next1 = (int(target_index) < int(sentence.size())-2) ? postags[target_index+2] : "<s>";

    if (target_index > 0) {
	prev_pos = postags[target_index-1];
    } else {
	prev_pos = "<s>";
    }

    if (int(target_index) < int(sentence.size())-1) {
	next_pos = postags[target_index+1];
    } else {
	next_pos = "<s>";
    }
    return vector<feature_t>{sentence[target_index], prev, prev1, prev_pos, postags[target_index], next_pos, next1, next};
  }
}
*/
  // ==========================
  // ITERATION 2 - HYPOTHESIS 3 
  // ==========================
  const unsigned NUM_EPOCHS = 45;

  transducer_t your_classifier(const vector<sentence_t> &training_set, const vector<symbol_t> &postags, const vector<symbol_t> &iobes_tags) {

    frequent_token_collector vocab_collector;
    for (const auto &sentence:training_set) {
      for (const auto &token:sentence) {
        vocab_collector.add_occurence(token);
      }
    }
    vector<symbol_t> vocab = vocab_collector.list_frequent_tokens(1000);

    auto embedding_lookup = make_embedding_lookup(64, vocab);
    auto pos_embed = make_embedding_lookup(64, vocab);
    auto postag = make_onehot(postags);

    auto concatenate = make_concatenate(10);

    auto dense1 = make_dense_feedfwd(iobes_tags.size(), make_softmax());

    auto onehot_inverse = make_onehot_inverse(iobes_tags);

    return compose(group(embedding_lookup, postag, postag, postag, postag, postag, postag, postag, postag, postag), concatenate, dense1, onehot_inverse);
  }

  vector<feature_t> get_features(const vector<symbol_t> &sentence, const vector<symbol_t> &postags, unsigned target_index) {
    feature_t sub = (int(target_index) < int(sentence.size())-4) ? postags[target_index+4] : "<s>";
    feature_t add = (target_index > 4) ? postags[target_index-4] : "<s>";
    feature_t next_pos;
    feature_t prev_pos;
    feature_t next = (target_index > 3) && (int(postags.size()) > 3) ? postags[target_index-3] : "<s>";
    feature_t prev = (int(target_index) < int(postags.size())-3) ? postags[target_index+3] : "<s>";

    feature_t prev1 = (target_index > 2) ? postags[target_index-2] : "<s>";
    feature_t next1 = (int(target_index) < int(sentence.size())-2) ? postags[target_index+2] : "<s>";

    if (target_index > 0) {
	prev_pos = postags[target_index-1];
    } else {
	prev_pos = "<s>";
    }

    if (int(target_index) < int(sentence.size())-1) {
	next_pos = postags[target_index+1];
    } else {
	next_pos = "<s>";
    }
    return vector<feature_t>{sentence[target_index], sub, prev, prev1, prev_pos, postags[target_index], next_pos, next1, next, add};
  }
}
