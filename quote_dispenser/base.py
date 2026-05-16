class QuoteCard:	#명언 카드 하나
	def __init__(self, 
			  idx_q: int = 9999, 
			  tag: list = None, 
			  author: str = None, 
			  #view: int = 0,
			  #favorite: bool = False,
			  content: dict = None,
			  registered_by: str = "admin"):
		self.idx_q = idx_q
		self.tag = tag
		self.author = author
		#self.view = view
		#self.favorite = favorite	
		self.content = content
		if self.content is None:
			self.content = {"kor": None, "eng": None}
		self.registered_by = registered_by

import random
class Dispenser:
	def __init__(self, filepath:str, quote_cards: list = None):
		self.quote_cards = quote_cards
		if self.quote_cards is None:
			self.quote_cards = self.load_quote_cards(self.filepath)

	def load_quote_cards(self,
					  filepath: str
					  # tag: list = None, 	#필터링
					  # author: str = None, 
					  # view: int = 0,
					  # favorite: bool = False,
					  # content: dict = None,
					  # registered_by: str = "admin"
					  ):
		#quote.md 읽어서 QuoteCard들의 리스트 만들기
		return [] # QuoteCard들의 리스트

	def draw_quote_card(self):
		return random.choice(self.quote_cards)

	def search_quote_card(self, 
					   idx_q: int =0
					   # tag: list = None, 	#필터링
					   # author: str = None, 
					   # view: int = 0,
					   # favorite: bool = False,
					   # content: dict = None,
					   # registered_by: str = "admin"
					  ):
		for quote_card in self.quote_cards:
			if quote_card.idx_q == idx_q:
				return quote_card
		
		
		return None
