import streamlit as st
import requests

url_path="http://localhost:8001/posts"
if "page" not in st.session_state:
    st.session_state["page"] = url_path 

def go_to_detail(post_id):
    st.session_state["page"] = url_path+"/post/"+str(post_id)

def go_to_posts():
    st.session_state["page"] = url_path 

def plus_like(post_id):
    requests.patch(url_path+"/post/"+str(post_id)+"/like")
    st.rerun()


if st.session_state["page"]==url_path:
    response = requests.get("http://localhost:8001/posts")
    posts=response.json()

    st.title("게시판")
    for post in posts:
        st.button(post["title"], on_click=go_to_detail, args=(post["post_id"],))
        st.text(post["content"])
        st.text(post["author"])
        st.text(post["likes"])
        st.text(post["created_at"])

else:
    response=requests.get(st.session_state["page"])
    post=response.json()
    st.button("돌아가기",on_click=go_to_posts)
    st.title(post["title"])
    st.text(post["content"])
    st.text(post["author"])
    st.button("좋아요: "+str(post["likes"]),on_click=plus_like, args=(post["post_id"],))
    st.text(post["created_at"])
