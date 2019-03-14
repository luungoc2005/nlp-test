from newspaper import Article, Config
from tqdm import tqdm

urls = []
with open('news_urls.txt', 'r') as in_file:
    urls = in_file.readlines()

downloaded_urls = []
with open('downloaded_urls.txt', 'r') as downloaded_file:
    downloaded_urls = downloaded_file.readlines()

config = Config()
config.memoize_articles = False
config.fetch_images = False
config.language = 'vi'
config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:10.0) Gecko/20100101 Firefox/10.0'

errors = 0
count = 0
with open('news_corpus.txt', 'a') as out_file:
    with open('downloaded_urls.txt', 'a') as downloaded_file:
        for url in tqdm(list(set(urls))):
            if url not in downloaded_urls:
                try:
                    article = Article(url=url, language='vi', fetch_images=False, memoize_articles=False, \
                        browser_user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:10.0) Gecko/20100101 Firefox/10.0')
                    article.download()
                    article.parse()

                    downloaded_file.write(url + '\n')
                    out_file.write(article.title + '\n\n' + article.text + '\n\n')
                    count += 1
                except:
                    errors += 1

        downloaded_file.close()
    out_file.close()

print('%s articles downloaded, %s errors' % (str(count), str(errors)))