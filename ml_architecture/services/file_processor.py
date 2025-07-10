import PyPDF2
from docx import Document
from fastapi import UploadFile
import io
import pytesseract
from pdf2image import convert_from_bytes
import tempfile
import os
from PIL import Image

# ĐẢM BẢO RẰNG TESSERACT-OCR ĐÃ ĐƯỢC CÀI ĐẶT VÀ THÊM VÀO BIẾN MÔI TRƯỜNG PATH CỦA HỆ THỐNG
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Hạn chế sử dụng hardcode

async def process_cv_file(file: UploadFile) -> str:
    """
    Xử lý file CV (PDF hoặc DOCX) và trả về nội dung dạng text
    """
    try:
        content = await file.read()
        
        # Kiểm tra kích thước file
        if len(content) == 0:
            raise ValueError("File trống")
            
        # Kiểm tra định dạng file
        if not file.filename:
            raise ValueError("Tên file không hợp lệ")
            
        if file.filename.endswith('.pdf'):
            return extract_text_from_pdf(content)
        elif file.filename.endswith('.docx'):
            return extract_text_from_docx(content)
        else:
            raise ValueError("Định dạng file không được hỗ trợ. Chỉ chấp nhận file PDF hoặc DOCX")
            
    except Exception as e:
        raise Exception(f"Lỗi khi xử lý file: {str(e)}")

def extract_text_from_pdf(content: bytes) -> str:
    """
    Trích xuất text từ file PDF.
    Ưu tiên trích xuất trực tiếp. Nếu không hiệu quả (ra ít text), mới dùng OCR.
    """
    try:
        # Luôn thử trích xuất trực tiếp trước tiên vì nó nhanh và chính xác cho PDF dạng text
        print("Thông tin: Bắt đầu trích xuất text trực tiếp...")
        text_direct = extract_text_direct(content)
        
        # Đặt ra một ngưỡng hợp lý. Nếu một CV có ít hơn 200 ký tự,
        # khả năng cao đó là file scan hoặc file có vấn đề.
        MIN_CHARS_FOR_DIRECT_EXTRACTION = 200

        if len(text_direct.strip()) >= MIN_CHARS_FOR_DIRECT_EXTRACTION:
            print(f"Thông tin: Trích xuất trực tiếp thành công ({len(text_direct.strip())} ký tự). Bỏ qua OCR.")
            return text_direct
            
        # Nếu trích xuất trực tiếp không hiệu quả, thử với OCR
        print(f"Thông tin: Trích xuất trực tiếp chỉ được {len(text_direct.strip())} ký tự (dưới ngưỡng {MIN_CHARS_FOR_DIRECT_EXTRACTION}). Thử với OCR...")
        text_ocr = extract_text_ocr(content)

        # Nếu OCR có kết quả, trả về kết quả đó.
        if text_ocr and len(text_ocr.strip()) > 0:
            print(f"Thông tin: OCR thành công ({len(text_ocr.strip())} ký tự).")
            return text_ocr
        
        # Nếu OCR thất bại nhưng trích xuất trực tiếp có một ít nội dung, trả về nội dung đó
        if text_direct.strip():
            print("Cảnh báo: OCR không thành công hoặc không có nội dung. Trả về kết quả trích xuất trực tiếp (có thể không đầy đủ).")
            return text_direct

        # Nếu cả hai phương pháp đều thất bại
        raise Exception("Không thể trích xuất bất kỳ nội dung nào từ file PDF. File có thể trống, được bảo vệ, hoặc là file ảnh không có chữ.")
        
    except Exception as e:
        raise Exception(f"Lỗi khi đọc file PDF: {str(e)}")

def extract_text_direct(content: bytes) -> str:
    """
    Trích xuất text trực tiếp từ PDF
    """
    try:
        pdf_file = io.BytesIO(content)
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        
        if pdf_reader.is_encrypted:
            raise Exception("File PDF được bảo vệ bằng mật khẩu")
            
        if len(pdf_reader.pages) == 0:
            raise Exception("File PDF không có nội dung")
            
        text = ""
        total_pages = len(pdf_reader.pages)
        pages_with_text = 0
        
        for i, page in enumerate(pdf_reader.pages, 1):
            try:
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text += page_text + "\n"
                    pages_with_text += 1
            except Exception as e:
                print(f"Lỗi khi đọc trang {i}: {str(e)}")
                continue
                
        if pages_with_text < total_pages and pages_with_text > 0:
            print(f"Cảnh báo: Chỉ đọc được text trực tiếp từ {pages_with_text}/{total_pages} trang.")
            
        return text.strip()
        
    except Exception as e:
        print(f"Lỗi khi trích xuất text trực tiếp (direct): {str(e)}")
        return ""

def extract_text_ocr(content: bytes) -> str:
    """
    Trích xuất text từ PDF sử dụng OCR
    """
    try:
        # Thử phương pháp 1: Sử dụng pdf2image
        try:
            images = convert_from_bytes(content)
            text = ""
            for i, image in enumerate(images, 1):
                try:
                    page_text = pytesseract.image_to_string(image, lang='vie+eng')
                    if page_text and page_text.strip():
                        text += page_text + "\n"
                except Exception as e:
                    print(f"Lỗi OCR trang {i} (pdf2image): {str(e)}")
                    continue
            if text.strip():
                return text.strip()
        except Exception as e:
            print(f"Lỗi convert PDF sang ảnh (pdf2image): {str(e)}. Có thể do Poppler chưa được cài đặt hoặc không nằm trong PATH.")

        # Thử phương pháp 2: Sử dụng PyMuPDF (fitz)
        try:
            import fitz
            pdf_document = fitz.open(stream=content, filetype="pdf")
            text = ""
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                pix = page.get_pixmap(matrix=fitz.Matrix(300/72, 300/72))  # 300 DPI
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                page_text = pytesseract.image_to_string(img, lang='vie+eng')
                if page_text and page_text.strip():
                    text += page_text + "\n"
            pdf_document.close()
            if text.strip():
                return text.strip()
        except Exception as e:
            print(f"Lỗi convert PDF sang ảnh (PyMuPDF): {str(e)}")

        # Nếu cả hai phương pháp đều thất bại
        return ""
    except Exception as e:
        print(f"Lỗi chung trong quá trình OCR: {str(e)}")
        return ""

def extract_text_from_docx(content: bytes) -> str:
    """
    Trích xuất text từ file DOCX
    """
    try:
        docx_file = io.BytesIO(content)
        doc = Document(docx_file)
        text = ""
        
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
            
        return text.strip()
    except Exception as e:
        raise Exception(f"Lỗi khi đọc file DOCX: {str(e)}") 