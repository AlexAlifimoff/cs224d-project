from nltk.tokenize import StanfordTokenizer

class TextVectorizer(object):
    def __init__(self, params, mapping = None):
        self.params = params
        if mapping is None:
            self.mapping = {}
        else:
            self.mapping = mapping

        self.tokenizer = StanfordTokenizer() 


    def vectorize(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        print(tokenized_text)

if __name__ == "__main__":
    params = {}
    tv = TextVectorizer(params)

    input = "An atom is the smallest constituent unit of ordinary matter that has the properties \
            of a chemical element. Every solid, liquid, gas and plasma is composed of neutral or \
            ionized atoms"
    summary = "An atom is the smallest unit of matter"

    tv.vectorize(input)

    tv.vectorize(summary)

