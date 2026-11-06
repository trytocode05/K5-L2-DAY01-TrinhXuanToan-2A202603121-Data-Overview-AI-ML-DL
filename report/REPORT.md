Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn
Runtime Colab: CPU

Python / PyTorch / Ultralytics: 3.13.18 / 2.11.0+cpu / 8.4.145

Checkpoint: yolo11n-cls.pt, yolo11n.pt, yolo11n-seg.pt

Ngưỡng detection/segmentation dùng khi tạo JSON: 0.35 (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

Thay đổi so với notebook nguồn: Không ghi nhận thay đổi; các phiên bản, checkpoint và ngưỡng trong output khớp cấu hình notebook.
Cách dùng mẫu này: Báo cáo được điền từ output notebook và hai JSON đã chạy. Notebook không tự tạo file này; upload REPORT.md vào Colab Files trước khi chạy ô đóng ZIP. Giải nén bộ ZIP và đưa các nội dung vào report/ trên GitHub, rồi nộp link repo trên LMS. Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output.



1. Phân loại ảnh
Mở file: day1_lab_outputs/classification_predictions.json và visuals/classification_top5.png.
Chỉ dùng ảnh sample_id = traffic, lấy dòng rank = 1.
1. Record hạng 1: class_id=`468`; class_name=`cab`; rank=`1`; score=`0.510915`; taxonomy_name=`ImageNet-1K`.
2. Record này nói về cả bức ảnh, không phải một vật thể riêng. Checkpoint classification xếp hạng lớp cho toàn bộ ảnh traffic; nó không tạo một box riêng cho từng xe hay người trong ảnh.
3. Danh sách lớp đến từ taxonomy đi kèm checkpoint, ở đây là ImageNet-1K; model không tự nghĩ ra tên lớp tùy ý trong lúc dự đoán.
4. Cần ghi cả class_id, class_name và taxonomy_name để xác định lớp rõ ràng: ID là mã số, tên lớp giúp con người đọc, còn taxonomy cho biết ID/tên đó thuộc bộ lớp nào. Một ID có thể mang ý nghĩa khác trong taxonomy khác.
5. Guideline cần định nghĩa tiêu chí chọn một nhãn đại diện cho toàn ảnh: chủ thể/chủ đề chính là gì, xử lý thế nào khi ảnh có nhiều nhóm vật thể, và quy tắc ưu tiên khi có nhiều chủ thể nổi bật hoặc không có chủ thể trội. Annotator phải áp dụng cùng một tiêu chí cho mọi ảnh.
6. Không lấy score làm ground truth vì đó chỉ là mức độ model tự tin với prediction. Model có thể tự tin nhưng vẫn dự đoán sai; ground truth phải dựa trên ảnh và guideline do con người áp dụng, không dựa vào điểm confidence.
2. Phát hiện vật thể
Mở file: day1_lab_outputs/detection_predictions.json và visuals/detection_predictions.png.
Chỉ dùng ảnh sample_id = kitchen.
1. Một record thực tế:
   - class_name=`person`
   - score=`0.912625`
   - bbox_xyxy=`[385.33, 69.24, 498.92, 348.92]`
   - bbox_width=`113.58`
   - bbox_height=`279.68`
2. Box được đo bằng pixel, gốc tọa độ ở góc trên-trái ảnh. Với record trên, góc trên-trái là (385.33, 69.24), góc dưới-phải là (498.92, 348.92). Hộp rộng 113.58 pixel và cao 279.68 pixel.
3. Số prediction trên ảnh kitchen:
   - Ngưỡng 0.20: 17 prediction.
   - Ngưỡng 0.60: 6 prediction.
   - (Để đối chiếu, ngưỡng lưu trong JSON là 0.35: 11 prediction.)
4. Khi hạ ngưỡng, giữ lại nhiều hộp có score thấp hơn. Reviewer cần kiểm tra thêm từng hộp để phát hiện prediction không đúng vật thể, hộp trùng, lớp sai hoặc box bao quá rộng/hẹp; đồng thời không được mặc định rằng mọi prediction score thấp đều sai.
5. Quy tắc box đề xuất: vẽ hộp chữ nhật bao sát phần vật thể nhìn thấy, với (x_min, y_min) là góc trên-trái và (x_max, y_max) là góc dưới-phải; dùng tọa độ pixel, hạn chế nền và không gộp vật thể lân cận vào cùng box. Với vật bị cắt bởi mép ảnh, chỉ bao phần nhìn thấy theo quy tắc thống nhất.
6. Ở mép trái ảnh kitchen có một người chỉ xuất hiện một phần; JSON có prediction person với box bắt đầu tại x=0.12 pixel. Guideline nên quy định cách xử lý vật thể bị cắt mép: nếu vẫn nhận diện chắc chắn thì gán nhãn phần nhìn thấy; nếu chỉ còn mảnh nhỏ và không đủ căn cứ xác định lớp/ranh giới thì dừng và escalation tới Lab Coach, không tự suy đoán.
3. Phân đoạn theo từng đối tượng
Mở file: day1_lab_outputs/segmentation_predictions.json và visuals/segmentation_prediction.png.
Chỉ dùng ảnh sample_id = kitchen.
1. Một record thực tế:
   - instance_id=`kitchen-001`
   - class_name=`person`
   - score=`0.899318`
   - polygon_point_count=`348`
   - 5 điểm đầu của polygon_xy=`[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]`
2. Box chỉ cho biết hình chữ nhật bao quanh vật thể. Polygon cho biết đường biên không đều của từng instance, bám sát hình dạng và phần pixel nhìn thấy, nhờ đó thể hiện hình dạng chi tiết hơn và ít bao gồm nền hơn.
3. instance_id là mã để phân biệt từng instance trong output của ảnh/lần chạy này, ví dụ kitchen-001. Hai vật thể cùng lớp vẫn cần ID và polygon riêng. ID này không mô tả loại lớp (đó là class_id/`class_name`) và không theo dõi cùng một vật thể qua các khung hình/thời gian (tracking ID).
4. Quy tắc mask đề xuất: vẽ polygon dọc theo ranh giới pixel nhìn thấy của đúng một instance, không ăn sang nền hoặc vật thể bên cạnh; giữ riêng từng vật cùng lớp. Với phần bị che khuất, chỉ gán phần thực sự nhìn thấy theo guideline, không tự vẽ phần ẩn.
5. Ảnh kitchen có cây ở góc trên-trái bị cắt bởi cả mép trên và mép trái; mask potted plant cũng đi sát biên ảnh. Guideline cần nói rõ chỉ vẽ phần nhìn thấy đến mép khung hình. Ở những vùng ranh giới không rõ vì vật thể chạm/che nhau (ví dụ các vật trên mặt bàn), annotator nên hỏi reviewer/Lab Coach; nếu không thể xác định biên đáng tin cậy thì escalation thay vì tự tạo đường biên.
4. Vòng đời và kiểm tra chất lượng
Lab hôm nay chỉ xem prediction; các prediction dưới đây không phải ground truth đã được xác nhận. Chuỗi production là:

ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework
Tác vụ	Ground truth trông như thế nào? (đơn vị nhãn)	Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON	Annotator làm gì?	Reviewer xem gì?
Phân loại ảnh	Một class_id/`class_name` cho toàn bộ ảnh, theo taxonomy đã chọn.	Ảnh traffic có nhiều loại phương tiện; top-1 của ImageNet-1K là cab (score 0.510915), các hạng sau gồm minibus và police_van. Đây là prediction, chưa chứng minh nhãn đúng.	Dùng tiêu chí “chủ đề/chủ thể chính” trong guideline để chọn một lớp; không sao chép máy móc top-1 hoặc score. Escalate nếu ảnh có nhiều chủ đề ngang nhau.	Kiểm tra lớp có đúng taxonomy không, tiêu chí chọn nhãn có nhất quán không và ảnh có đủ căn cứ cho một nhãn duy nhất không.
Phát hiện vật thể	Một nhãn lớp và một box xyxy cho mỗi vật thể đủ điều kiện.	Người ở mép trái ảnh kitchen bị cắt; số prediction thay đổi từ 17 ở threshold 0.20 xuống 6 ở 0.60.	Gán từng vật thể nhìn thấy, dùng box bao sát, xử lý vật cắt mép theo guideline; không dùng threshold của model làm quy tắc bỏ qua vật thể.	Kiểm tra vật bị bỏ sót, box trùng, nhầm lớp, box lệch/bao nền và tính nhất quán ở vật bị cắt/che khuất.
Instance segmentation	Một class_id/`class_name`, instance_id và polygon pixel riêng cho mỗi instance.	Cây ở góc trên-trái bị cắt mép; polygon của dining table có 598 điểm, cho thấy biên phức tạp; vật trên mặt bàn chạm/che nhau.	Vẽ mask theo phần nhìn thấy, giữ riêng từng instance cùng lớp; hỏi khi không xác định được biên ở vùng chồng lấn.	Kiểm tra mask có lấn nền/vật khác không, có bỏ sót phần nhìn thấy không, đường biên có hợp lý không và instance_id có duy nhất không.
