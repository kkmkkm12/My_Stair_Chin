import os


def rename_image_files(directory_path):
    # 디렉토리 내의 이미지 파일 목록을 얻어옵니다.
    image_files = [file for file in os.listdir(directory_path) if
                   file.lower().endswith(('.jpg'))]

    # 파일들을 순회하며 이름 변경
    for index, filename in enumerate(image_files, start=206):
        # 새 파일 이름 생성
        extension = os.path.splitext(filename)[1]  # 확장자
        new_filename = f'img{index:02}{extension}'

        # 파일 이름 변경
        os.rename(os.path.join(directory_path, filename), os.path.join(directory_path, new_filename))
        print(f"{filename}을(를) {new_filename}으로 변경했습니다.")


# 변경할 파일들이 들어있는 디렉토리 경로
directory_path = './rename_image'

# 함수 호출
rename_image_files(directory_path)
