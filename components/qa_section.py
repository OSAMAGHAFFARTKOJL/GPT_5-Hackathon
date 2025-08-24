import streamlit as st

def render_qa_section():
    """Render the Q&A section for repository questions"""
    # Q&A Section - Always available if repository is analyzed
    if st.session_state.repo_analyzed and st.session_state.analyzer:
        st.subheader("❓ Ask Questions About the Repository")
        st.markdown("You can ask questions about the code, request explanations, or ask for help with errors/issues.")
        
        # Question input
        question = st.text_input(
            "Enter your question:", 
            placeholder="e.g., 'How does the authentication work?', 'Fix this error: ModuleNotFoundError', 'Explain the main function'"
        )
        
        # Submit button for questions
        if st.button("💡 Get Answer") and question:
            with st.spinner("🔍 Searching through code and generating answer..."):
                answer = st.session_state.analyzer.answer_question(question)
                
                st.subheader("🤖 Answer")
                st.write(answer)
                
                # Add to chat history
                if 'qa_history' not in st.session_state:
                    st.session_state.qa_history = []
                
                st.session_state.qa_history.append({
                    'question': question,
                    'answer': answer
                })
        
        # Display recent Q&A history
        if st.session_state.qa_history:
            with st.expander("📜 Recent Questions & Answers"):
                for i, qa in enumerate(reversed(st.session_state.qa_history[-5:])):  # Show last 5
                    st.write(f"**Q{len(st.session_state.qa_history)-i}:** {qa['question']}")
                    st.write(f"**A:** {qa['answer'][:200]}{'...' if len(qa['answer']) > 200 else ''}")
                    st.write("---")