# nlp_project

## Проект: natural language processing

### План проекта

- Создаём git-репозиторий `nlp_project`
- Разрабатываем [multipage](https://blog.streamlit.io/introducing-multipage-apps/)-приложение с использованием [streamlit](streamlit.io):
   - Классификация отзыва на фильм на английском языке
      - Поле ввода для пользовательского отзыва
      - Результаты предсказаний класса (позитивный/негативный) тремя моделями:
         - Классический ML-алгоритм, обученный на BagOfWords/TF-IDF представлении
         - RNN/LSTM модель
         - BERT
   - Генерация текста GPT-моделью по пользовательскому prompt:
      - Пользователь может регулировать длину выдаваемой последовательности
      - Число генераций
      - Температуру или top-k/p
   - Произвольная задача:
      - Саммаризация текста: пользователь вводит большой текст, модель делает саммари (huggingface → models → summarization)
      - Ответ на вопрос: пользователь вводит вопрос и контекст (в котором есть ответ) – модель пытается ответить на вопрос (huggingface → model → question answering)

__Датасеты__: [link](https://drive.google.com/drive/folders/1o4Uzyt9I-pR4TBmo_Tmc_z0QGRXG7nKZ?usp=sharing)
