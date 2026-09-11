# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Khi temperature tăng từ 0.0 lên 1.5, phản hồi thường trở nên phong phú hơn, đa dạng hơn và ít lặp lại hơn. Ở temperature thấp, model chọn câu trả lời ổn định, gần với câu nói quen thuộc; khi temperature cao, câu trả lời có xu hướng sáng tạo hơn nhưng cũng dễ lệch khỏi trọng tâm hoặc mơ hồ hơn. Nói ngắn gọn, temperature càng cao thì độ ngẫu nhiên và tính sáng tạo càng tăng, nhưng độ chắc chắn và sự nhất quán giảm đi.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Tôi sẽ chọn temperature khoảng 0.2 đến 0.5. Lý do là chatbot hỗ trợ khách hàng cần câu trả lời rõ ràng, nhất quán, ít sai lệch và phù hợp với chính sách nội bộ của doanh nghiệp. Ở mức này, model vẫn có đủ độ mượt mà để diễn đạt tự nhiên nhưng không quá sáng tạo đến mức trả lời lệch khỏi mục tiêu hoặc tạo ra thông tin thiếu tin cậy. Chỉ khi cần trải nghiệm trò chuyện thân thiện hơn, tôi mới tăng nhẹ nhiệt độ chứ không dùng mức cao.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Với 10.000 người dùng mỗi ngày, mỗi người gọi 3 lần, tổng số lượt là 30.000 lượt/ngày. Nếu mỗi lượt sinh ra 350 token đầu ra thì tổng output là 10.500.000 token/ngày. Theo bảng giá, GPT-4o chi phí 0.010 USD/1K token trong phần output, còn GPT-4o-mini là 0.0006 USD/1K token. Vậy chi phí hàng ngày ước tính là 105 USD cho GPT-4o và 6.3 USD cho GPT-4o-mini, tức GPT-4o đắt hơn khoảng 16.7 lần. GPT-4o xứng đáng với chi phí khi xử lý các câu hỏi phức tạp, cần tư duy chuyên sâu hoặc tạo ra phản hồi chất lượng cao cho khách hàng doanh nghiệp; còn mini phù hợp cho FAQ, hỗ trợ nhanh, trả lời ngắn, hoặc xử lý khối lượng lớn với chi phí thấp.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Hai phản hồi sẽ khác nhau rõ rệt về độ dài, phong cách từ vựng và mức độ ví dụ. Với persona là giáo viên tiểu học, câu trả lời sẽ ngắn, dễ hiểu, dùng ví dụ gần gũi và ít thuật ngữ chuyên môn; còn với persona chuyên gia tài chính, câu trả lời sẽ dài hơn, chuyên sâu hơn và có nhiều thuật ngữ kỹ thuật như “ledger”, “smart contract”, “rủi ro”. System prompt ảnh hưởng rất lớn vì nó “định hướng” hành vi của model: model sẽ ưu tiên phong cách, mức độ chi tiết và cách giải thích phù hợp với vai trò mà ta giao cho nó.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Nếu đoạn văn tiếng Việt khoảng 100 từ, số token theo tiktoken thường cao hơn ước lượng “số từ / 0.75”, và chênh lệch thường dao động khoảng 20–50% tùy vào câu cụ thể. Nguyên nhân là tokenizer không đo theo số từ đơn thuần, mà dựa trên cách máy chia văn bản thành đơn vị ngữ nghĩa, và tiếng Việt thường có nhiều từ ghép, dấu câu, ký tự đặc biệt và cấu trúc khúc mắc hơn so với tiếng Anh. Vì vậy, cùng độ dài, tiếng Việt thường tốn nhiều token hơn tiếng Anh khi muốn diễn đạt cùng một ý nghĩa.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming rất quan trọng khi người dùng cần cảm giác phản hồi ngay lập tức, như trong chatbot, trợ lý ảo, hỗ trợ khách hàng hoặc ứng dụng học tập có nhịp độ tương tác cao. Với streaming, người dùng thấy chữ hiện dần và cảm giác tương tác tốt hơn, giảm độ chờ và tăng sự tin tưởng. Non-streaming lại phù hợp hơn khi nội dung dài, cần tổng hợp hoàn chỉnh trước khi hiển thị, hoặc khi hệ thống cần tối ưu chi phí và hiệu suất hơn là trải nghiệm “đến chữ liên tục”.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff giúp giảm tải đồng thời lên server khi API đang quá tải, vì mỗi lần retry sẽ chờ lâu hơn dần theo cấp số nhân, thay vì tất cả client cùng phát tín hiệu vào lúc như nhau. Nếu hàng nghìn client cùng retry với delay cố định, họ sẽ “đập” vào máy chủ cùng một thời điểm, tạo ra hiện tượng storm retry, làm tình trạng quá tải tăng mạnh hơn và dễ khiến toàn bộ dịch vụ bị treo. Backoff phân tán thời gian retry, giúp hệ thống có cơ hội phục hồi và giảm nguy cơ nghẽn mạng.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona là “trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt và giải thích rõ từng bước”. System prompt mẫu: “Bạn là trợ giảng thân thiện của khóa AI, trả lời ngắn gọn bằng tiếng Việt, giải thích rõ ràng từng bước bằng ví dụ thực tế và tránh ngôn ngữ quá kỹ thuật.” Hai từ ngữ quan trọng là “trả lời ngắn gọn” và “bằng tiếng Việt”. “Trả lời ngắn gọn” giúp model tập trung vào ý chính, tránh dài dòng không cần thiết; “bằng tiếng Việt” đảm bảo câu trả lời phù hợp với người dùng Việt Nam, tự nhiên và dễ hiểu hơn. Tôi cũng nhấn mạnh “giải thích rõ từng bước” để câu trả lời vừa ngắn vừa có cấu trúc dễ học.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất của trợ lý hiện tại là thiếu bộ nhớ dài hạn và lịch sử hội thoại quá ngắn. Khi người dùng hỏi nhiều lượt, model dễ quên ngữ cảnh quan trọng từ những cuộc trò chuyện trước đó, khiến phản hồi mất tính liên kết. Một cải thiện cụ thể là lưu toàn bộ hội thoại vào database/session store, rồi mỗi lần gọi API chỉ gửi một “context tóm tắt” + các tin nhắn gần đây nhất. Cách này giúp hệ thống vẫn có ngữ cảnh dài hạn mà không làm prompt quá dài và tốn chi phí.

---

## Danh Sách Kiểm Tra Nộp Bài

- [ ] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [ ] Cả 4 checkpoint pytest đều pass
- [ ] Tất cả 9 câu trong file này đã được trả lời
- [ ] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
