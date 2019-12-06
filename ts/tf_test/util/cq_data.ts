import * as fs from "fs";
import * as path from "path";

import * as pngjs from "pngjs";

export class Image {
    readonly pixel: Pixel[][];

    constructor(public readonly buffer: Buffer, public readonly width, public readonly height) {
        this.pixel = [];

        for (let x = 0; x < width; ++x) {
            this.pixel.push([]);
            const arr = this.pixel[this.pixel.length - 1];

            for (let y = 0; y < height; ++y) {
            }
        }
    }
}

export class Pixel {
    get r() {
        return this.buffer[0];
    }

    get g() {
        return this.buffer[1];
    }

    get b() {
        return this.buffer[2];
    }

    get a() {
        return this.buffer[3];
    }

    set r(value: number) {
        this.buffer[0] = value;
    }

    set g(value: number) {
        this.buffer[1] = value;
    }

    set b(value: number) {
        this.buffer[2] = value;
    }

    set a(value: number) {
        this.buffer[3] = value;
    }

    constructor(public readonly buffer: Buffer) {

    }
}

export enum CqDataType {
    Face,
    Whole,
    SD
}

export class CqData {
    readonly images: Buffer[];

    constructor(public readonly type: CqDataType) {
        this.images = [];

        let dir = path.join(__dirname, "..", "..", "cq_data", "cq");

        switch (type) {
        case CqDataType.Face:
            {
                dir = path.join(dir, "face");
            }
            break;

        case CqDataType.Whole:
            {
                dir = path.join(dir, "whole");
            }
            break;

        case CqDataType.SD:
            {
                dir = path.join(dir, "sd");
            }
            break;
        }

        const fileNames = fs.readdirSync(dir);

        fileNames.forEach(fileName => {
            const fullPath = path.join(dir, fileName);

            const png = new pngjs.PNG();

            fs.createReadStream(fullPath)
                .pipe(png)
                .on("parsed",
                    buffer => {
                        const originalBuffer = png.data;

                        const numChannel = png.data.length / (png.width * png.height);

                        for (let x = 0; x < png.width; ++x) {
                            for (let y = 0; y < png.height; ++y) {

                            }
                        }
                    });

        });
    }
}