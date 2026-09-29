from collections import Counter

class BytePairEncoding:
    def __init__(self, corpus : str, vocabulary_size:int=10, space_symbole:str="␣")-> None:
        self.result = [space_symbole if caractere == " " else caractere for caractere in corpus]
        self.vocabulary_size = vocabulary_size

        # Méthode pour obtenir la paire la plus présente dans la liste après fusion
        self.get_most_present_pair = lambda token_list : Counter(token_list).most_common(1)[0][0]

        # Méthode pour créer des paires entre tous les éléments d'une liste
        self.get_paired_list = lambda token_list : [token_list[i]+token_list[i+1] for i in range(len(token_list)-1)]

    @staticmethod
    def merge_pair(token_list: list[str], pair_str: str):
        new_list = []
        i = 0
        while i < len(token_list):
            if i < len(token_list) - 1 and token_list[i] + token_list[i + 1] == pair_str:
                new_list.append(pair_str)
                i += 2  # on saute les deux tokens fusionnés
            else:
                new_list.append(token_list[i])
                i += 1
        return new_list

    def compute_vocabulary(self):
        while len(set(self.result)) > self.vocabulary_size:
            paired = self.get_paired_list(self.result)
            if not paired:
                break
            best_pair = self.get_most_present_pair(paired)
            new_result = self.merge_pair(self.result, best_pair)
            if new_result == self.result:  # plus aucune fusion possible
                break
            self.result = new_result

        self.result = set(self.result)
bpe = BytePairEncoding("le chat mange le chat dort le chat joue", vocabulary_size=10)
bpe.compute_vocabulary()

print(bpe.result)
