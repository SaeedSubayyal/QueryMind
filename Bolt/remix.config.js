/** @type {import('@remix-run/dev').AppConfig} */
export default {
  ignoredRouteFiles: ["**/.*"],
  future: {
    v3_lazyRouteDiscovery: true,
    v3_singleFetch: true
  }
}; 