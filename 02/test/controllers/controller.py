from modules.DB_module import get_connection

def get_posts():
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM posts")       # 전체 게시글 가져오는 SQL
    posts = cursor.fetchall()
    connection.close()
    return posts

def get_post(post_id):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("SELECT * FROM posts WHERE post_id = %s", (post_id,))
    post = cursor.fetchone()
    connection.close()
    return post

def like_post(post_id):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute("UPDATE posts SET likes = likes + 1 WHERE post_id = %s", (post_id,))
    connection.commit()       
    connection.close()