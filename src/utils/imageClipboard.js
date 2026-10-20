/** Convert a rendered plot source into a Blob without fetching data: URLs.
 *
 * Electron applies connect-src to fetch(). Decoding a local data URL directly
 * keeps the CSP narrow while allowing the user-facing Copy Image action.
 */
export async function imageSourceToBlob(source) {
  if (!source || typeof source !== 'string') {
    throw new Error('No image source is available to copy.');
  }
  if (!source.startsWith('data:')) {
    const response = await fetch(source);
    if (!response.ok) throw new Error(`Could not load image (${response.status}).`);
    return response.blob();
  }

  const separator = source.indexOf(',');
  if (separator < 0) throw new Error('Malformed image data URL.');
  const header = source.slice(0, separator);
  const payload = source.slice(separator + 1);
  const mimeType = /^data:([^;,]+)/i.exec(header)?.[1] || 'image/png';
  const bytes = header.toLowerCase().includes(';base64')
    ? Uint8Array.from(atob(payload), char => char.charCodeAt(0))
    : new TextEncoder().encode(decodeURIComponent(payload));
  return new Blob([bytes], { type: mimeType });
}

export async function copyImageToClipboard(source) {
  if (!navigator.clipboard?.write || typeof ClipboardItem === 'undefined') {
    throw new Error('Image clipboard support is unavailable in this desktop runtime.');
  }
  const blob = await imageSourceToBlob(source);
  await navigator.clipboard.write([new ClipboardItem({ [blob.type]: blob })]);
}
