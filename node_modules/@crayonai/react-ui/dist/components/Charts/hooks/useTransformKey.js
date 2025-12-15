import { useMemo, useRef } from "react";
export const useTransformedKeys = (keys) => {
    // Use a ref to maintain a stable cache of key -> UUID mappings
    const cacheRef = useRef({});
    return useMemo(() => {
        return keys.reduce((acc, key) => {
            // Only generate a new UUID if we don't already have one for this key
            if (!cacheRef.current[key]) {
                cacheRef.current[key] = crypto.randomUUID();
            }
            acc[key] = cacheRef.current[key];
            return acc;
        }, {});
    }, [keys]);
};
//# sourceMappingURL=useTransformKey.js.map