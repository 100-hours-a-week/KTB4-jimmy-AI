import base
import argparse

d=base.Dispenser()

parser = argparse.ArgumentParser(
                    prog='Quote Dispenser',
                    description='Can draw quotes',
                    epilog="""
                    python cli.py   :명언카드 하나 뽑기
                    python cli.py n :n개 뽑기                    
                    python cli.py --idx idx :인덱스로 명어카드 찾기
                    python cli.py --all :전체 카탈로그 보기
                    python cli.py --help    :도움말 출력
                    """)

parser.add_argument('n', nargs='?', default=1, type=int)
parser.add_argument('--idx', type=int)
parser.add_argument('--all', action='store_true')
parser.add_argument('--help', action='store_true')