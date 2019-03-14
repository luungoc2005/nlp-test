import newspaper
from newspaper import Config

news_sites = [
    'https://vnexpress.net/',
    'https://thanhnien.vn/',
    'https://news.zing.vn/',
    'http://kenh14.vn/',
    'https://www.24h.com.vn/',
    'http://soha.vn/',
    'https://saostar.vn/',
    'https://tuoitre.vn/',
    'http://congan.com.vn/',
    'https://plo.vn/',
    'http://baophapluat.vn/',
    'http://www.doisongphapluat.com/',
    'https://anninhthudo.vn/',
    'https://www.phunuonline.com.vn/',
    'https://baomoi.com/',
    'http://hoahoctro.vn/',
    'http://muctim.com.vn/',
    'http://echip.com.vn/',
    'http://www.khoahocphothong.com.vn/',
    # 'https://trandaiquang.org/',
    'https://vietnamnet.vn/',
    'https://dantri.com.vn/',
    'http://baodatviet.vn/',
    'http://vietbao.vn/',
    'http://docbao.vn/',
    'https://www.nguoi-viet.com/',
    'https://www.bbc.com/vietnamese/vietnam',
    'http://cafef.vn/',
    'https://vietnamfinance.vn/',
    'http://vneconomy.vn/tai-chinh.htm',
    'http://tapchitaichinh.vn/',
    'http://danviet.vn/',
    'https://vietbao.com/',
    'https://tintuconline.com.vn/',
    'https://baodautu.vn/',
    'http://www.sggp.org.vn/',
    'http://www.nhandan.com.vn/',
    'https://vtv.vn/',
    'https://laodong.vn/',
    'http://bongdaplus.vn/',
    'http://www.qdnd.vn/',
    'http://kinhtedothi.vn/',
    'http://cand.com.vn/',
    'https://vnanet.vn/',
    'http://www.baobinhphuoc.com.vn/',
    'http://baobinhdinh.com.vn/',
    'http://baoangiang.com.vn/',
    'http://thegioivanhoa.com.vn/',
    'http://baobinhduong.vn/',
    'http://www.baobaclieu.vn/',
    'http://www.baobackan.org.vn/',
    'http://baobacninh.com.vn/',
    'https://vn.yahoo.com/',
    'http://baolamdong.vn/',
    'https://tintaynguyen.com/topic/da-lat/',

    'https://toidicodedao.com/',
    'https://triethocduongpho.net/',
]
existing_urls = []
with open('news_urls.txt', 'r') as infile:
    existing_urls = infile.readlines()
    infile.close()
existing_urls = [url.strip() for url in existing_urls]
print('Found %s URLs previously crawled' % str(len(existing_urls)))

urls = []
# config = Config()
# config.memoize_articles = False
# config.fetch_images = False
# config.language = 'vi'
# config.browser_user_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:10.0) Gecko/20100101 Firefox/10.0'

for site in news_sites:
    news_site = newspaper.build(site, language='vi', fetch_images=False, memoize_articles=False, \
        browser_user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X x.y; rv:10.0) Gecko/20100101 Firefox/10.0')
    count = 0
    added_count = 0
    for article in news_site.articles:
        current_url = article.url.strip()
        if current_url not in existing_urls:
            urls.append(current_url)
            added_count += 1
        count += 1
    print('%s - %s articles, %s added' % (site, count, added_count))

with open('news_urls.txt', 'a') as outfile:
    for url in urls:
        outfile.write(url + '\n')
    outfile.close()