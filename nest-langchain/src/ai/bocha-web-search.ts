import z from 'zod';

const bochaWebPageSchema = z.object({
  name: z.string().optional(),
  url: z.string().optional(),
  summary: z.string().optional(),
  snippet: z.string().optional(),
  siteName: z.string().optional(),
  siteIcon: z.string().optional(),
  dateLastCrawled: z.string().optional(),
});

const bochaWebSearchResponseSchema = z.object({
  code: z.number(),
  msg: z.string().nullish(),
  data: z
    .object({
      webPages: z
        .object({
          value: z.array(bochaWebPageSchema).optional(),
          values: z.array(bochaWebPageSchema).optional(),
        })
        .optional(),
    })
    .optional(),
});

type BochaWebPage = z.infer<typeof bochaWebPageSchema>;

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function formatWebPage(page: BochaWebPage, idx: number): string {
  return `引用: ${idx + 1}
标题: ${page.name ?? ''}
URL: ${page.url ?? ''}
摘要: ${page.summary ?? page.snippet ?? ''}
网站名称: ${page.siteName ?? ''}
网站图标: ${page.siteIcon ?? ''}
发布时间: ${page.dateLastCrawled ?? ''}`;
}

export async function runBochaWebSearch(
  apiKey: string,
  baseUrl: string,
  query: string,
  count: number,
): Promise<string> {
  const response = await fetch(baseUrl, {
    method: 'POST',
    body: JSON.stringify({
      query,
      count,
      freshness: 'noLimit',
      summary: true,
    }),
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
  });

  if (!response.ok) {
    return `搜索 API 请求失败，状态码: ${response.status}, 错误信息: ${await response.text()}`;
  }

  let json: unknown;
  try {
    json = await response.json();
  } catch (error: unknown) {
    return `搜索结果解析失败：${errorMessage(error)}`;
  }

  const parsed = bochaWebSearchResponseSchema.safeParse(json);
  if (!parsed.success) {
    return `搜索结果格式无效：${parsed.error.message}`;
  }

  const { code, msg, data } = parsed.data;
  if (code !== 200 || !data) {
    return `搜索 API 返回失败：${msg ?? '未知错误'}`;
  }

  const webpages = data.webPages?.value ?? data.webPages?.values ?? [];
  if (!webpages.length) {
    return '未找到相关结果。';
  }

  return webpages.map(formatWebPage).join('\n\n');
}
