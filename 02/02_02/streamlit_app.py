import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="커뮤니티 게시판", page_icon="📋", layout="centered")

# 세션 상태 초기화
if "page" not in st.session_state:
    st.session_state.page = "list"
if "selected_post_id" not in st.session_state:
    st.session_state.selected_post_id = None
if "summary" not in st.session_state:
    st.session_state.summary = None


def go_to_list():
    st.session_state.page = "list"
    st.session_state.selected_post_id = None
    st.session_state.summary = None


def go_to_detail(post_id):
    st.session_state.page = "detail"
    st.session_state.selected_post_id = post_id
    st.session_state.summary = None


# ── 게시글 목록 페이지 ──────────────────────────────────────────────────────
if st.session_state.page == "list":
    st.title("📋 커뮤니티 게시판")

    # 새 게시글 작성 폼
    with st.expander("✏️ 새 게시글 작성"):
        with st.form("create_post"):
            title = st.text_input("제목")
            content = st.text_area("내용")
            submitted = st.form_submit_button("작성")
            if submitted:
                if title and content:
                    res = requests.post(f"{API_URL}/posts", json={"title": title, "content": content})
                    if res.status_code == 200:
                        st.success("게시글이 작성되었습니다!")
                        st.rerun()
                else:
                    st.warning("제목과 내용을 모두 입력해주세요.")

    st.divider()

    # 게시글 목록
    try:
        posts = requests.get(f"{API_URL}/posts").json()
    except requests.exceptions.ConnectionError:
        st.error("FastAPI 서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.")
        st.stop()

    if not posts:
        st.info("게시글이 없습니다. 첫 번째 글을 작성해보세요!")
    else:
        for post in reversed(posts):  # 최신 글이 위로
            col1, col2 = st.columns([5, 1])
            with col1:
                st.subheader(post["title"])
                st.caption(f"👍 {post['likes']}  💬 {len(post['comments'])}개")
            with col2:
                st.write("")
                if st.button("보기", key=f"view_{post['id']}"):
                    go_to_detail(post["id"])
                    st.rerun()
            st.divider()


# ── 게시글 상세 페이지 ──────────────────────────────────────────────────────
elif st.session_state.page == "detail":
    post_id = st.session_state.selected_post_id

    try:
        post = requests.get(f"{API_URL}/posts/{post_id}").json()
    except requests.exceptions.ConnectionError:
        st.error("FastAPI 서버에 연결할 수 없습니다.")
        st.stop()

    if st.button("← 목록으로"):
        go_to_list()
        st.rerun()

    st.title(post["title"])
    st.write(post["content"])
    st.divider()

    # 좋아요 & AI 요약 버튼
    col1, col2 = st.columns(2)
    with col1:
        if st.button(f"👍 좋아요  {post['likes']}"):
            requests.patch(f"{API_URL}/posts/{post_id}")
            st.rerun()
    with col2:
        if st.button("🤖 AI 요약 생성"):
            with st.spinner("Gemma4가 요약 중... (잠시 기다려주세요)"):
                try:
                    res = requests.get(f"{API_URL}/posts/{post_id}/summary", timeout=120)
                    st.session_state.summary = res.json().get("summary", "요약 실패")
                except requests.exceptions.Timeout:
                    st.session_state.summary = "요약 시간이 초과됐습니다."

    # 요약 결과 표시 (버튼 클릭 후에도 유지)
    if st.session_state.summary:
        st.info(f"**AI 요약**\n\n{st.session_state.summary}")

    st.divider()

    # 댓글 목록
    comments = post.get("comments", [])
    st.subheader(f"💬 댓글 {len(comments)}개")
    if comments:
        for comment in comments:
            st.markdown(f"- {comment['content']}")
    else:
        st.caption("아직 댓글이 없습니다.")

    # 댓글 작성
    with st.form("create_comment"):
        comment_content = st.text_input("댓글을 입력하세요")
        if st.form_submit_button("댓글 작성") and comment_content:
            requests.post(
                f"{API_URL}/posts/{post_id}/comments",
                json={"content": comment_content},
            )
            st.rerun()
