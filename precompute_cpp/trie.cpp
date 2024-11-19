#include <iostream>
#include <fstream>
#include <map>
#include <string>
#include <memory>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

class TrieNode {
public:
    std::map<std::string, int> children;
    bool isEndOfWord;

    TrieNode() : isEndOfWord(false) {}

    // Serialize the children map to a JSON string
    std::string serialize() const {
        json nodeData = json::object();
        for (const auto &child : children) {
            nodeData[child.first] = child.second;
        }
        return nodeData.dump();
    }

    void deserialize(const std::string &data) {
        json nodeData = json::parse(data);
        for (auto &[key, value] : nodeData.items()) {
            children[key] = value;
        }
    }
};

class Trie {
public:
    std::shared_ptr<TrieNode> root;

    Trie() {
        root = std::make_shared<TrieNode>();
    }

     //Save trie to disk
    void save(const std::string& filename) {
        std::ofstream outFile(filename, std::ios::binary);
        if (outFile.is_open()) {
            outFile << root->serialize();
            outFile.close();
        } else {
            std::cerr << "Error opening file for saving." << std::endl;
        }
    }

     //Load trie from disk
    void load(const std::string& filename) {
        std::ifstream inFile(filename, std::ios::binary);
        if (inFile.is_open()) {
            std::stringstream buffer;
            buffer << inFile.rdbuf();
            root->deserialize(buffer.str());
            inFile.close();
        } else {
            std::cerr << "Error opening file for loading." << std::endl;
        }
    }

     //Add a word to the trie
    void insert(const std::string& word, int value) {
        auto current = root;
        for (char c : word) {
            std::string key(1, c);
            if (current->children.find(key) == current->children.end()) {
                current->children[key] = value;
            }
            current = std::make_shared<TrieNode>();
        }
        current->isEndOfWord = true;
    }
    int query(const std::string& word) {
        auto current = root;
        for (char c : word) {
            std::string key(1, c);
             //If the current node doesn't have the child for this character
            if (current->children.find(key) == current->children.end()) {
                return -1; // Word not found
            }
            current = std::make_shared<TrieNode>();  // move to the next child node
        }
         //If we reach the end of the word and it’s marked as a valid end of word, return the associated value
        return current->isEndOfWord ? current->children.begin()->second : -1; // or a default value
    }
};

int main() {
    Trie trie;
    trie.insert("examp2le", 41);
    trie.insert("test", 10);

    std::cout << trie.query("test");
     //Save the trie to disk
    trie.save("trie_data.json");

     //Create a new Trie instance and load the data
    Trie newTrie;
    newTrie.load("trie_data.json");
    std::cout << newTrie.query("test");

    return 0;
}
