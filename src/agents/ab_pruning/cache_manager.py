import time
import pickle

class CacheManager:
    def __init__(self, cache_path, time_update=60, time_threshold=120, size_threshold=1e6):
        self.time_update = time_update
        self.time_threshold = time_threshold
        self.size_threshold = size_threshold
        self.num_updates = 0
        self.last_save_time = time.time()
        self.cache_path = cache_path
        self.load_cache()

    def update_cache(self, encoded_state, best_action, best_value):
        if not encoded_state in self.cache:
            self.cache[encoded_state] = {
                "a": best_action,
                "v": best_value
            }
            self.num_updates += 1
        self.check_save_conditions()
    
    def check_save_conditions(self):
        if self.num_updates == 0:
            return
        time_elapsed = time.time() - self.last_save_time
        if time_elapsed > self.time_threshold:
            print(f"After {time_elapsed} seconds, saving cache.")
            self.save_cache()

    # def get_cache_size(self):
    #     return len(str(self.cache).encode('utf-8'))

    def save_cache(self):
        start_save = time.time()
        with open(self.cache_path, 'wb') as f:
            pickle.dump(self.cache, f)
        time_taken = time.time() - start_save
        print(f"Cache saved with {len(self.cache)} items in {time_taken} seconds at {self.cache_path}")
        self.num_updates = 0
        self.last_save_time = time.time()

    def load_cache(self):
        try:
            with open(self.cache_path, 'rb') as f:
                self.cache = pickle.load(f)
            print(f"Cache loaded with {len(self.cache)} items.")
        except Exception as e:
            print(e)
            print("Could not load cache.")
            self.cache = {}

    def retrieve_action(self, encoded_state):
        if encoded_state not in self.cache:
            return None
        if encoded_state not in self.cache:
            return None
        if "a" in self.cache[encoded_state]:
            return self.cache[encoded_state]["a"]
        raise ValueError("Action not found in cache")
    
    def retrieve_value(self, encoded_state):
        if encoded_state not in self.cache:
            return None
        if "v" in self.cache[encoded_state]:
            return self.cache[encoded_state]["v"]
        raise ValueError("Value not found in cache")