//
// Created by Dekai WU and YAN Yuchen on 20190304.
//

#ifndef _TOKEN_FEEDFORWARD_CLASSIFIER_HPP_
#define _TOKEN_FEEDFORWARD_CLASSIFIER_HPP_

#include <unordered_set>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_set>

/**
 * a model that takes a token and predicts a label
 */
class token_feedforward_classifier {
  std::string id_m;
public:
  token_feedforward_classifier() = delete;
  token_feedforward_classifier(const token_feedforward_classifier&) = delete;
  token_feedforward_classifier(token_feedforward_classifier&&) noexcept = default;
  token_feedforward_classifier &operator=(const token_feedforward_classifier&) = delete;
  token_feedforward_classifier &operator=(token_feedforward_classifier&&) noexcept = default;
  /**
   * constructs the model
   * \param vocab the set of all possible tokens
   * \param embedding_size word embedding size
   * \param num_hidden_layers number of hidden layers in between
   * \param labels the set of all possible labels
   */
  token_feedforward_classifier(const std::vector<std::string> &vocab,
                              unsigned embedding_size,
                              unsigned num_hidden_layers,
                              const std::vector<std::string> &labels);

  /**
   * given a training set, train the model
   * \param training_set the set of all training tokens
   * \param training_oracles the desired label of the training tokens
   * \param num_epochs number of iterations to train on the training set
   * \return an aggregated loss for each epoch
   */
  std::vector<float> train(const std::vector<std::string> &training_set,
                           const std::vector<std::string> &training_oracles,
                           unsigned num_epochs);

  /**
   * given a test set, and predict their labels
   * \param test_set
   * \return the predicted label
   */
  std::vector<std::string> test(const std::vector<std::string> &test_set) const;
};

// below are utility functions that you don't need in assignment 1

/**
 * read dataset into list of tokens and oracles
 * \param is input stream of dataset
 * \return * list of tokens
 *         * list of oracles
 */
std::pair<std::vector<std::string>, std::vector<std::string>> read_dataset(std::istream& is);

/**
 * remove duplicated items in an array
 * does not preserve the original order
 * \param arr input array
 * \return array with duplicated items removed
 */
std::vector<std::string> unique(const std::vector<std::string>& arr);

double compute_accuracy(const std::vector<std::string>& hypothesis, const std::vector<std::string>& oracle);

#endif
