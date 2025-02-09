import scrapy

class ZNUCrawler(scrapy.Spider):
    name = "znu_crawler" 
    allowed_domains = ["znu.ac.ir"]  
    start_urls = ["https://www.znu.ac.ir/"]  
    
    custom_settings = {
        "FEED_FORMAT": "json",  
        "FEED_URI": "znu_data.json", 
        "AUTOTHROTTLE_ENABLED": True,  
        "ROBOTSTXT_OBEY": True,  
        "DOWNLOAD_DELAY": 0.5, 
        "CLOSESPIDER_PAGECOUNT": 12000,  
        "DEPTH_LIMIT": 5,  
    }

    def parse(self, response):
        text_data = " ".join(response.css("p::text, h1::text, h2::text, h3::text").getall()).strip()

        if text_data: 
            yield {
                "url": response.url,
                "text": text_data
            }

        for next_page in response.css("a::attr(href)").getall():
            if next_page.startswith("/") or "znu.ac.ir" in next_page:
                yield response.follow(next_page, callback=self.parse)

