css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #e8e8e8
}
.chat-message.bot {
    background-color: #7686ad
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 60px;
  max-height: 60px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <div style="width: 60px; height: 60px; border-radius: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 32px; box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);">🤖</div>
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <div style="width: 60px; height: 60px; border-radius: 10px; background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); display: flex; align-items: center; justify-content: center; color: white; font-weight: bold; font-size: 32px; box-shadow: 0 4px 12px rgba(245, 87, 108, 0.3);">👤</div>
    </div>    
    <div class="message" style="color: #333;">{{MSG}}</div>
</div>
'''