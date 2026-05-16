import random	#draw_quote_cards


class QuoteCard:	#명언 카드 하나
	def __init__(self, 
			  idx: int = 9999, 
			  tag: list = None, 
			  author: str = None, 
			  #view: int = 0,
			  #favorite: bool = False,
			  content: dict = None,
			  registered_by: str = "admin"):
		self.idx = idx
		self.tag = tag
		self.author = author
		#self.view = view
		#self.favorite = favorite	
		self.content = content
		if self.content is None:
			self.content = {"kor": None, "eng": None}
		self.registered_by = registered_by

class Dispenser:
	def __init__(self, filepath:str="./catalog.md", quote_cards: list = None):
		self.quote_cards = quote_cards
		if self.quote_cards is None:
			self.quote_cards = self.load_quote_cards() 
	def load_quote_cards(self,
					  filepath: str ="./catalog.md"
					  # tag: list = None, 	#필터링은 db에 넣고
					  # author: str = None, 
					  # view: int = 0,
					  # favorite: bool = False,
					  # content: dict = None,
					  # registered_by: str = "admin"
					  ):
	#quote.md 읽어서 QuoteCard들의 리스트 만들기
		# 1. 파일 전체를 읽는다
		with open(filepath, "r") as f:
			text=f.read()
		# 2. "---"를 기준으로 나눈다 → 블록 리스트
		blocks=text.split("---")
		blocks=[b for b in blocks if b.strip()]

		# 3. 각 블록을 줄 단위로 나눈다
		blocks = [b.split("\n") for b in blocks]
		blocks=[[b for b in block if b.strip()] for block in blocks]

		# 4. 각 줄에서 "키: 값"을 추출한다
		for b_idx, block in enumerate(blocks):
			blocks[b_idx]={b.split(":",1)[0].strip():b.split(":",1)[1].strip() for b in block}
		#print(blocks)
			blocks[b_idx]["idx"]=int(blocks[b_idx]["idx"])
			blocks[b_idx]["tags"]=blocks[b_idx]["tags"].split(',')
			blocks[b_idx]["tags"]=[b.strip() for b in blocks[b_idx]["tags"]]
		# 5. QuoteCard를 만들어서 리스트에 추가한다
			blocks[b_idx]=QuoteCard(
				idx=blocks[b_idx]["idx"],
				tag=blocks[b_idx]["tags"],
				author=blocks[b_idx]["author"],
				#view=blocks[b_idx]["view"],
				#favorite=blocks[b_idx]["favorite"],
				content={"kor":blocks[b_idx]["kor"],"eng":blocks[b_idx]["eng"]},
				registered_by=blocks[b_idx]["registered_by"]
				)
		print(blocks)
		# 6. 리스트를 반환한다
		return blocks # QuoteCard들의 리스트

	def draw_quote_card(self):
		return random.choice(self.quote_cards)

	def search_quote_card(self, 	#이미 load된 덱에서 찾기
					   idx: int =0
					   # tag: list = None, 	#필터링은 db에 넣고
					   # author: str = None, 
					   # view: int = 0,
					   # favorite: bool = False,
					   # content: dict = None,
					   # registered_by: str = "admin"
					  ):
		for quote_card in self.quote_cards:
			if quote_card.idx == idx:
				return quote_card
		

		return None
